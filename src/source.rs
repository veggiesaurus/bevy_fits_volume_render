//! The viewer's only way to the voxels. Requests go out and their events come back on a later frame, so nothing
//! else reads the cube. The source is fits-core's engine: natively in-process, its jobs on the async compute pool,
//! and in the browser in a Web Worker (web.rs); or a server (remote.rs). It wakes the app when there's something to
//! collect.

pub use crate::fits::engine::{CubeletId, Event, Mode, Request};
#[cfg(target_arch = "wasm32")]
use crate::web::WorkerSource as LocalSource;
#[cfg(not(target_arch = "wasm32"))]
use local::DataSource as LocalSource;

pub enum DataSource {
    /// A file read here, or in the browser by its worker.
    Local(LocalSource),
    Remote(crate::remote::Source),
}

impl DataSource {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn local(cube: crate::fits::Cube) -> Self {
        Self::Local(local::DataSource::local(cube))
    }

    pub fn send(&self, request: Request) {
        match self {
            Self::Local(source) => source.send(request),
            Self::Remote(source) => source.send(request),
        }
    }

    /// The events that have arrived since the last call.
    pub fn drain(&self) -> Vec<Event> {
        match self {
            Self::Local(source) => source.drain(),
            Self::Remote(source) => source.drain(),
        }
    }

    /// A server has no noise measures or masks to give, nor processed cubelets.
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod local {
    use std::{
        collections::HashSet,
        sync::{Arc, Mutex},
    };

    use bevy::{
        platform::time::Instant,
        prelude::*,
        tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
    };

    use super::{CubeletId, Event, Request};
    use crate::{
        fits::{Cube, engine::Engine},
        wake,
    };

    /// Jobs at a time, so that a cubelet asked for first is cut first.
    const IN_FLIGHT: usize = 32;

    pub struct DataSource {
        state: Mutex<State>,
        answered: Arc<Mutex<Vec<Event>>>,
    }

    struct State {
        engine: Engine,
        /// Dropping a task cancels it, if it hasn't started.
        running: Vec<(Option<CubeletId>, Task<()>)>,
    }

    impl DataSource {
        pub fn local(cube: Cube) -> Self {
            Self {
                state: Mutex::new(State {
                    engine: Engine::new(cube),
                    running: Vec::new(),
                }),
                answered: default(),
            }
        }

        pub fn send(&self, request: Request) {
            let mut state = self.state.lock().unwrap();
            if let Request::RemoveCubelets { ids } = &request {
                let ids: HashSet<&CubeletId> = ids.iter().collect();
                state
                    .running
                    .retain(|(id, _)| id.is_none_or(|id| !ids.contains(&id)));
            }
            if let Some(event) = state.engine.request(request) {
                self.answered.lock().unwrap().push(event);
                wake();
            }
            self.pump(&mut state);
        }

        /// The events that have arrived since the last call.
        pub fn drain(&self) -> Vec<Event> {
            self.pump(&mut self.state.lock().unwrap());
            std::mem::take(&mut *self.answered.lock().unwrap())
        }

        /// Starts what can be started, in the order asked. Finished tasks are polled first, so that a panic in one
        /// comes out here.
        fn pump(&self, state: &mut State) {
            state
                .running
                .retain_mut(|(_, task)| block_on(poll_once(task)).is_none());
            while state.running.len() < IN_FLIGHT
                && let Some(job) = state.engine.next_job()
            {
                let (cubelet, answered) = (job.cubelet(), self.answered.clone());
                let task = AsyncComputeTaskPool::get().spawn(async move {
                    let start = Instant::now();
                    let event = job.run();
                    let seconds = start.elapsed().as_secs_f32();
                    if seconds >= 0.1 {
                        // Cuts of the overview from level 0 are slow on a big cube, and there are a lot of them.
                        if cubelet.is_some() {
                            debug!("{job} in {seconds:.2} s");
                        } else {
                            info!("{job} in {seconds:.2} s");
                        }
                    }
                    // Before the wake, so that the engine sees it's done.
                    drop(job);
                    answered.lock().unwrap().extend(event);
                    // Even with nothing to send: what waited on this job may be able to start now.
                    wake();
                });
                state.running.push((cubelet, task));
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::{
            RwLock,
            atomic::{AtomicUsize, Ordering},
        };

        use bevy::tasks::TaskPool;

        use super::*;
        use crate::{fits, source::Mode};

        #[test]
        fn removed_cubelets_never_arrive_and_the_rest_do() {
            let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
            let dims = [512, 512, 4];
            let mut bytes: Vec<u8> = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]
                .into_iter()
                .zip(["T", "-32", "3", "512", "512", "4"])
                .flat_map(|(key, value)| format!("{key:<8}= {value:>20}{:50}", "").into_bytes())
                .chain(format!("{:<80}", "END").into_bytes())
                .collect();
            bytes.resize(2880, b' ');
            bytes.extend((0..512 * 512 * 4).flat_map(|i| (i as f32).to_be_bytes()));
            bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
            let source = DataSource::local(fits::read_fits(&bytes[..], None).unwrap());
            let chain = fits::levels(dims, 384, 2048).chain;
            assert_eq!(chain, [[1; 3], [2, 2, 1], [4, 4, 1], [8, 8, 1]]);
            source.send(Request::Levels {
                factors: chain,
                chain: 4,
            });

            // Every thread of the pool waits at the gate, so that nothing sent runs before the removals.
            let gate = Arc::new(RwLock::new(()));
            let closed = gate.write().unwrap();
            let waiting = Arc::new(AtomicUsize::new(0));
            for _ in 0..pool.thread_num() {
                let (gate, waiting) = (gate.clone(), waiting.clone());
                pool.spawn(async move {
                    waiting.fetch_add(1, Ordering::Relaxed);
                    drop(gate.read());
                })
                .detach();
            }
            while waiting.load(Ordering::Relaxed) < pool.thread_num() {
                std::thread::yield_now();
            }

            // The Mean ones wait on a level job, which takes one of the IN_FLIGHT, and level 0 fills the rest.
            let mean: Vec<_> = (0..16)
                .map(|i| CubeletId::new(Mode::Mean, default(), 1, [i % 4, i / 4, 0]))
                .collect();
            let full: Vec<_> = (0..64)
                .map(|i| CubeletId::new(Mode::Max, default(), 0, [i % 8, i / 8, 0]))
                .collect();
            source.send(Request::AddCubelets {
                ids: [&mean[..], &full[..]].concat(),
            });
            assert_eq!(source.state.lock().unwrap().running.len(), IN_FLIGHT);
            // Two spawned but held at the gate, one still queued and one waiting on the pyramid.
            let removed = [full[0], full[1], full[63], mean[15]];
            source.send(Request::RemoveCubelets {
                ids: removed.to_vec(),
            });
            drop(closed);

            let mut arrived = Vec::new();
            let start = Instant::now();
            while !source.state.lock().unwrap().running.is_empty() || arrived.len() < 76 {
                assert!(start.elapsed().as_secs() < 10, "{} of 76", arrived.len());
                arrived.extend(source.drain().into_iter().filter_map(|event| match event {
                    Event::Cubelet { id, .. } => Some(id),
                    _ => None,
                }));
                std::thread::yield_now();
            }
            arrived.sort_by_key(|id| (id.level, id.y, id.x));
            let mut wanted: Vec<_> = (mean.iter().chain(&full))
                .filter(|id| !removed.contains(id))
                .copied()
                .collect();
            wanted.sort_by_key(|id| (id.level, id.y, id.x));
            assert_eq!(arrived, wanted);
        }
    }
}
