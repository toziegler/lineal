#![warn(
    clippy::all,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    //clippy::cargo
)]
#![allow(clippy::must_use_candidate)]
use std::{
    collections::HashMap,
    fs::File,
    io::{stdout, BufWriter, Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    time::{Duration, Instant},
};

use bitflags::Flags;
use perf_event_open_sys::{
    bindings::{
        perf_event_attr, perf_hw_id, perf_type_id, PERF_COUNT_HW_BRANCH_MISSES,
        PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_MISS, PERF_COUNT_SW_TASK_CLOCK, PERF_FORMAT_TOTAL_TIME_ENABLED,
        PERF_FORMAT_TOTAL_TIME_RUNNING,
    },
    perf_event_open,
};

#[derive(Debug)]
#[repr(C)]
#[derive(Default)]
struct ReadFormat {
    value: u64,
    time_enabled: u64,
    time_running: u64,
}

#[derive(Debug)]
struct Event {
    pe: perf_event_open_sys::bindings::perf_event_attr,
    file: File,
    prev: ReadFormat,
    data: ReadFormat,
}

impl Event {
    pub(crate) fn read_counter(&self) -> f64 {
        let multiplex_correction = (self.data.time_enabled - self.prev.time_enabled) as f64
            / (self.data.time_running - self.prev.time_running) as f64;
        (self.data.value - self.prev.value) as f64 * multiplex_correction
    }
}
#[derive(Debug)]
pub struct PerfEvents {
    events: Vec<Event>,
    names: Vec<String>,
    begin: Instant,
    end: Instant,
}

bitflags::bitflags! {
    #[derive(Clone, Copy)]
    pub struct EventDomain: u64 {
        const USER = 0b1;
        const KERNEL = 0b10;
        const HYPERVISOR = 0b100;
    }
}

impl PerfEvents {
    pub fn new() -> Result<Self, std::io::Error> {
        let mut names = Vec::new();
        let mut perf_event_atrrs = Vec::new();
        let mut events = Vec::new();
        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "cycles",
            perf_event_open_sys::bindings::PERF_TYPE_HARDWARE,
            perf_event_open_sys::bindings::PERF_COUNT_HW_CPU_CYCLES,
            EventDomain::all(),
        ));
        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "kcycles",
            perf_event_open_sys::bindings::PERF_TYPE_HARDWARE,
            perf_event_open_sys::bindings::PERF_COUNT_HW_CPU_CYCLES,
            EventDomain::KERNEL,
        ));
        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "instructions",
            perf_event_open_sys::bindings::PERF_TYPE_HARDWARE,
            perf_event_open_sys::bindings::PERF_COUNT_HW_INSTRUCTIONS,
            EventDomain::all(),
        ));

        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "L1-misses",
            perf_event_open_sys::bindings::PERF_TYPE_HW_CACHE,
            PERF_COUNT_HW_CACHE_L1D
                | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
            EventDomain::all(),
        ));

        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "LLC-misses",
            perf_event_open_sys::bindings::PERF_TYPE_HARDWARE,
            PERF_COUNT_HW_CACHE_MISSES,
            EventDomain::all(),
        ));

        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "branch-misses",
            perf_event_open_sys::bindings::PERF_TYPE_HARDWARE,
            PERF_COUNT_HW_BRANCH_MISSES,
            EventDomain::all(),
        ));

        perf_event_atrrs.push(Self::register_counter(
            &mut names,
            "task-clock",
            perf_event_open_sys::bindings::PERF_TYPE_SOFTWARE,
            PERF_COUNT_SW_TASK_CLOCK,
            EventDomain::all(),
        ));

        names.push("IPC".to_string());
        names.push("GHz".to_string());
        names.push("CPUs".to_string());
        names.push("runtime".to_string());

        for (id, event) in perf_event_atrrs.iter_mut().enumerate() {
            let fd = unsafe { perf_event_open(event, 0, -1, -1, 0) };
            if fd < 0 {
                eprintln!("Error opening counter {}", names[id]);
                return Err(std::io::Error::from_raw_os_error(fd));
            }
            events.push(Event {
                pe: *event,
                file: unsafe { File::from_raw_fd(fd) },
                prev: ReadFormat::default(),
                data: ReadFormat::default(),
            });
        }

        Ok(Self {
            events,
            names,
            begin: Instant::now(),
            end: Instant::now(),
        })
    }

    fn register_counter(
        names: &mut Vec<String>,
        name: &str,
        perf_type: perf_type_id,
        perf_id: perf_hw_id,
        domain: EventDomain,
    ) -> perf_event_attr {
        names.push(name.to_string());
        let mut pe = perf_event_attr::default();
        //let mut pe = perf_event_attr::default();
        pe.type_ = perf_type;
        pe.size = u32::try_from(std::mem::size_of::<perf_event_attr>()).expect("could not cast");
        pe.config = u64::from(perf_id);
        pe.set_disabled(1);
        pe.set_inherit(1);
        pe.set_inherit_stat(0);
        pe.set_exclude_user((domain & EventDomain::USER).is_empty() as u64);
        pe.set_exclude_kernel((domain & EventDomain::KERNEL).is_empty() as u64);
        pe.set_exclude_hv((domain & EventDomain::HYPERVISOR).is_empty() as u64);
        pe.read_format = u64::from(PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING);
        pe
    }

    pub fn start_measurement(&mut self) -> Result<(), std::io::Error> {
        let mut data = [0_u8; std::mem::size_of::<ReadFormat>()];
        for event in &mut self.events {
            unsafe {
                libc::ioctl(
                    event.file.as_raw_fd(),
                    u64::from(perf_event_open_sys::bindings::RESET),
                    0,
                )
            };
            unsafe {
                libc::ioctl(
                    event.file.as_raw_fd(),
                    u64::from(perf_event_open_sys::bindings::ENABLE),
                    0,
                )
            };
            let bytes_read = event.file.read(&mut data)?;
            if bytes_read != data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Could not read event",
                ));
            }
            // interpret event prev
            event.prev.value = u64::from_ne_bytes(data[0..8].try_into().unwrap());
            event.prev.time_enabled = u64::from_ne_bytes(data[8..16].try_into().unwrap());
            event.prev.time_running = u64::from_ne_bytes(data[16..24].try_into().unwrap());
        }
        self.begin = Instant::now();
        Ok(())
    }

    pub fn stop_measurement(&mut self) -> Result<(), std::io::Error> {
        let mut data = [0_u8; std::mem::size_of::<ReadFormat>()];
        for event in &mut self.events {
            let bytes_read = event.file.read(&mut data)?;
            if bytes_read != data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Could not read event",
                ));
            }

            unsafe {
                libc::ioctl(
                    event.file.as_raw_fd(),
                    u64::from(perf_event_open_sys::bindings::DISABLE),
                    0,
                )
            };
            // interpret event
            event.data.value = u64::from_ne_bytes(data[0..8].try_into().unwrap());
            event.data.time_enabled = u64::from_ne_bytes(data[8..16].try_into().unwrap());
            event.data.time_running = u64::from_ne_bytes(data[16..24].try_into().unwrap());
        }
        self.end = Instant::now();
        Ok(())
    }

    fn get_duration(&self) -> Duration {
        self.end.duration_since(self.begin)
    }

    fn get_ipc(&self) -> f64 {
        self.get_counter("instructions") / self.get_counter("cycles")
    }

    fn get_cpus(&self) -> f64 {
        self.get_counter("task-clock") / (self.get_duration().as_secs_f64() * 1e9)
    }

    fn get_ghz(&self) -> f64 {
        self.get_counter("cycles") / self.get_counter("task-clock")
    }

    fn get_counter_normalized(&self, name: &str, normalization_constant: f64) -> f64 {
        match name {
            "IPC" => self.get_ipc(),
            "CPUs" => self.get_cpus(),
            "GHz" => self.get_ghz(),
            "runtime" => self.get_duration().as_secs_f64(),
            _ => self.get_counter(name) / normalization_constant,
        }
    }
    fn get_counter(&self, name: &str) -> f64 {
        match name {
            "IPC" => return self.get_ipc(),
            "CPUs" => return self.get_cpus(),
            "GHz" => return self.get_ghz(),
            "runtime" => return self.get_duration().as_secs_f64(),
            _ => {}
        }

        for i in 0..self.events.len() {
            if self.names[i] == name {
                return self.events[i].read_counter();
            }
        }
        -1.0
    }
    fn normalized_iterator(&self, normalization_constant: f64) -> PerfEventsIter<'_> {
        PerfEventsIter::new(self, normalization_constant)
    }
}

pub struct PerfEventsIter<'a> {
    events: &'a PerfEvents,
    iter: core::slice::Iter<'a, String>,
    normalization_constant: f64,
}

impl<'a> PerfEventsIter<'a> {
    pub(crate) fn new(events: &'a PerfEvents, normalization_constant: f64) -> Self {
        let iterator = events.names.iter();
        Self {
            events,
            iter: iterator,
            normalization_constant,
        }
    }
}

impl Iterator for PerfEventsIter<'_> {
    type Item = (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(name) => Some((
                name.to_owned(),
                self.events
                    .get_counter_normalized(name, self.normalization_constant),
            )),
            None => None,
        }
    }
}

trait OutputStrategy {
    fn output_report(&self, events: &PerfEvents);
}

struct CSVOutput {
    normalization_constant: f64,
    benchmark_params: HashMap<String, String>,
}

impl CSVOutput {
    fn print_report_impl(
        &self,
        events: &PerfEvents,
        header_out: &mut String,
        data_out: &mut String,
        normalization_constant: f64,
    ) {
        for e in events.normalized_iterator(normalization_constant) {
            self.print_counter(header_out, data_out, e.0.as_str(), e.1, true);
        }
        self.print_counter(header_out, data_out, "scale", normalization_constant, false);
    }
    fn print_params(
        &self,
        header_out: &mut String,
        data_out: &mut String,
        name: &str,
        param_value: &str,
        add_comma: bool,
    ) {
        let width = name.len().max(param_value.len());
        header_out.push_str(&format!(
            "{:width$}{}",
            name,
            if add_comma { "," } else { "" },
            width = width
        ));
        data_out.push_str(&format!(
            "{:width$}{}",
            param_value,
            if add_comma { "," } else { "" },
            width = width
        ));
    }

    fn print_counter(
        &self,
        header_out: &mut String,
        data_out: &mut String,
        name: &str,
        counter_value: f64,
        add_comma: bool,
    ) {
        let width = name.len().max(format!("{counter_value:.2}").len());
        header_out.push_str(&format!(
            "{:width$}{}",
            name,
            if add_comma { "," } else { "" },
            width = width
        ));
        data_out.push_str(&format!(
            "{:width$.2}{}",
            counter_value,
            if add_comma { "," } else { "" },
            width = width
        ));
    }
}

impl OutputStrategy for CSVOutput {
    fn output_report(&self, events: &PerfEvents) {
        let mut header = String::new();
        let mut data = String::new();
        let mut out = BufWriter::new(stdout());
        for (name, counter_value) in &self.benchmark_params {
            self.print_params(
                &mut header,
                &mut data,
                name.as_str(),
                counter_value.as_str(),
                true,
            );
        }
        self.print_report_impl(events, &mut header, &mut data, self.normalization_constant);
        writeln!(out, "{header}").unwrap();
        writeln!(out, "{data}").unwrap();
    }
}

// pass in parameters
pub struct PerfEventBlock {
    events: PerfEvents,
    benchmark_params: HashMap<String, String>,
    normalization_constant: f64,
}

impl PerfEventBlock {
    pub fn new(
        benchmark_params: HashMap<String, String>,
        normalization_constant: f64,
    ) -> Result<Self, std::io::Error> {
        let mut events = PerfEvents::new()?;
        events.start_measurement()?;
        Ok(Self {
            events,
            benchmark_params,
            normalization_constant,
        })
    }
    pub fn set_normalization_constant(&mut self, normalization_constant: f64) {
        self.normalization_constant = normalization_constant;
    }
}
impl Drop for PerfEventBlock {
    fn drop(&mut self) {
        self.events
            .stop_measurement()
            .expect("Could not stop PerfEventBlock");
        let strategy = CSVOutput {
            normalization_constant: self.normalization_constant,
            benchmark_params: self.benchmark_params.clone(),
        };
        strategy.output_report(&self.events);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        arch::x86_64::_mm_pause,
        io::{stdout, BufWriter},
    };

    use super::*;

    #[test]
    fn it_works() {
        let mut lineal = PerfEvents::new().unwrap();
        lineal.start_measurement().unwrap();
        for _ in 0..10_000_000 {
            unsafe { _mm_pause() };
        }
        lineal.stop_measurement().unwrap();

        let mut benchmark_params = HashMap::new();
        benchmark_params.insert("Test".to_owned(), "Running".to_owned());

        let csv_outputer = CSVOutput {
            normalization_constant: 10_000_000.0,
            benchmark_params,
        };
        csv_outputer.output_report(&lineal);
    }

    #[test]
    fn block_works() {
        let mut benchmark_params = HashMap::new();
        benchmark_params.insert("Test".to_owned(), "Running".to_owned());
        let block = PerfEventBlock::new(benchmark_params, 10_000_000.0);
        for _ in 0..10_000_000 {
            unsafe { _mm_pause() };
        }
    }
}
