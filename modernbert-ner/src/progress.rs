//! Minimal ASCII progress bar for the CLI (no emoji, no extra dependency).

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

pub struct Progress {
    label: String,
    total: u64,
    done: u64,
    started: Instant,
    last_draw: Instant,
    /// Off when stderr is redirected, so logs stay free of carriage returns.
    interactive: bool,
}

fn hms(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

impl Progress {
    pub fn new(label: &str, total: u64) -> Self {
        let now = Instant::now();
        Self {
            label: label.to_string(),
            total: total.max(1),
            done: 0,
            started: now,
            last_draw: now - Duration::from_secs(1),
            interactive: std::io::stderr().is_terminal(),
        }
    }

    pub fn add(&mut self, n: u64) {
        self.done += n;
        if self.last_draw.elapsed() >= Duration::from_millis(100) {
            self.draw(false);
        }
    }

    pub fn finish(mut self) {
        self.done = self.done.max(self.total);
        self.draw(true);
        eprintln!();
    }

    fn draw(&mut self, force_full: bool) {
        self.last_draw = Instant::now();
        let frac = (self.done as f64 / self.total as f64).clamp(0.0, 1.0);
        let elapsed = self.started.elapsed();
        let rate = self.done as f64 / elapsed.as_secs_f64().max(1e-9);
        let eta = if rate > 0.0 && !force_full {
            hms(Duration::from_secs_f64(
                (self.total - self.done.min(self.total)) as f64 / rate,
            ))
        } else {
            hms(Duration::ZERO)
        };

        const WIDTH: usize = 32;
        let filled = (frac * WIDTH as f64).round() as usize;
        let bar: String = (0..WIDTH)
            .map(|i| if i < filled { '=' } else { ' ' })
            .collect();

        let line = format!(
            "{} [{}] {:5.1}%  {}/{} chars  {:.0} chars/s  elapsed {}  eta {}",
            self.label,
            bar,
            frac * 100.0,
            self.done,
            self.total,
            rate,
            hms(elapsed),
            eta
        );

        let mut err = std::io::stderr();
        if self.interactive {
            let _ = write!(err, "\r{line}");
        } else if force_full {
            let _ = write!(err, "{line}");
        } else {
            return;
        }
        let _ = err.flush();
    }
}
