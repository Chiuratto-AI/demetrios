//! Work-stealing scheduler for parallel compilation

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

/// A unit of work
pub trait Task: Send + 'static {
    /// Execute the task
    fn execute(self: Box<Self>);

    /// Priority (higher = more urgent)
    fn priority(&self) -> i32 {
        0
    }

    /// Estimated cost (for load balancing)
    fn estimated_cost(&self) -> usize {
        1
    }
}

/// Work-stealing deque for a single worker
struct WorkDeque {
    tasks: Mutex<VecDeque<Box<dyn Task>>>,
}

impl WorkDeque {
    fn new() -> Self {
        WorkDeque {
            tasks: Mutex::new(VecDeque::new()),
        }
    }

    /// Push a task to the back (owner's end)
    fn push(&self, task: Box<dyn Task>) {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push_back(task);
    }

    /// Pop a task from the back (owner's end)
    fn pop(&self) -> Option<Box<dyn Task>> {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.pop_back()
    }

    /// Steal a task from the front (thief's end)
    fn steal(&self) -> Option<Box<dyn Task>> {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.pop_front()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        let tasks = self.tasks.lock().unwrap();
        tasks.is_empty()
    }

    /// Number of tasks
    fn len(&self) -> usize {
        let tasks = self.tasks.lock().unwrap();
        tasks.len()
    }
}

/// Work-stealing scheduler
pub struct WorkStealingScheduler {
    /// Per-worker deques
    deques: Vec<Arc<WorkDeque>>,

    /// Global task queue for overflow
    global_queue: Mutex<VecDeque<Box<dyn Task>>>,

    /// Shutdown flag
    shutdown: AtomicBool,

    /// Number of active workers
    active_workers: AtomicUsize,

    /// Condvar for waking workers
    work_available: (Mutex<bool>, Condvar),

    /// Worker threads
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl WorkStealingScheduler {
    pub fn new(num_workers: usize) -> Arc<Self> {
        let deques: Vec<_> = (0..num_workers)
            .map(|_| Arc::new(WorkDeque::new()))
            .collect();

        let scheduler = Arc::new(WorkStealingScheduler {
            deques,
            global_queue: Mutex::new(VecDeque::new()),
            shutdown: AtomicBool::new(false),
            active_workers: AtomicUsize::new(0),
            work_available: (Mutex::new(false), Condvar::new()),
            workers: Mutex::new(Vec::new()),
        });

        // Start worker threads
        for i in 0..num_workers {
            let sched = Arc::clone(&scheduler);
            let handle = thread::spawn(move || worker_loop(i, sched));
            scheduler.workers.lock().unwrap().push(handle);
        }

        scheduler
    }

    /// Submit a task
    pub fn submit(&self, task: Box<dyn Task>) {
        // Find least loaded worker
        let min_idx = self
            .deques
            .iter()
            .enumerate()
            .min_by_key(|(_, d)| d.len())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.deques[min_idx].push(task);

        // Wake a worker
        self.notify_work();
    }

    /// Submit multiple tasks
    pub fn submit_batch(&self, tasks: Vec<Box<dyn Task>>) {
        // Distribute evenly
        for (i, task) in tasks.into_iter().enumerate() {
            let idx = i % self.deques.len();
            self.deques[idx].push(task);
        }

        self.notify_work();
    }

    /// Shutdown the scheduler
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.notify_work();
    }

    /// Wait for all tasks to complete
    pub fn wait_idle(&self) {
        loop {
            if self.active_workers.load(Ordering::SeqCst) == 0
                && self.deques.iter().all(|d| d.is_empty())
                && self.global_queue.lock().unwrap().is_empty()
            {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn notify_work(&self) {
        let (lock, cvar) = &self.work_available;
        let mut available = lock.lock().unwrap();
        *available = true;
        cvar.notify_all();
    }

    fn pop_global(&self) -> Option<Box<dyn Task>> {
        let mut q = self.global_queue.lock().unwrap();
        q.pop_front()
    }
}

fn worker_loop(id: usize, scheduler: Arc<WorkStealingScheduler>) {
    let my_deque = &scheduler.deques[id];

    loop {
        if scheduler.shutdown.load(Ordering::SeqCst) {
            break;
        }

        // Try to get work from own deque
        if let Some(task) = my_deque.pop() {
            scheduler.active_workers.fetch_add(1, Ordering::SeqCst);
            task.execute();
            scheduler.active_workers.fetch_sub(1, Ordering::SeqCst);
            continue;
        }

        // Try global queue
        if let Some(task) = scheduler.pop_global() {
            scheduler.active_workers.fetch_add(1, Ordering::SeqCst);
            task.execute();
            scheduler.active_workers.fetch_sub(1, Ordering::SeqCst);
            continue;
        }

        // Try to steal from others
        let mut stolen = false;
        for (i, deque) in scheduler.deques.iter().enumerate() {
            if i == id {
                continue;
            }

            if let Some(task) = deque.steal() {
                scheduler.active_workers.fetch_add(1, Ordering::SeqCst);
                task.execute();
                scheduler.active_workers.fetch_sub(1, Ordering::SeqCst);
                stolen = true;
                break;
            }
        }

        if stolen {
            continue;
        }

        // No work - wait or shutdown
        let (lock, cvar) = &scheduler.work_available;
        let mut available = lock.lock().unwrap();

        if scheduler.shutdown.load(Ordering::SeqCst) {
            break;
        }

        let result = cvar
            .wait_timeout(available, Duration::from_millis(10))
            .unwrap();
        available = result.0;
        *available = false;
    }
}

/// Statistics from scheduler
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub tasks_executed: usize,
    pub tasks_stolen: usize,
    pub total_wait_time_ms: u64,
    pub total_work_time_ms: u64,
}
