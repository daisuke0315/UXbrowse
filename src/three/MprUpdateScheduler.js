export class MprUpdateScheduler {
  constructor(callback, { minUpdateIntervalMs = 80 } = {}) {
    this.callback = callback;
    this.minUpdateIntervalMs = minUpdateIntervalMs;
    this.frameRequested = false;
    this.running = false;
    this.pending = false;
    this.lastRun = 0;
  }

  request(reason = "change") {
    this.pending = true;
    this.reason = reason;
    if (this.frameRequested) return;
    this.frameRequested = true;
    requestAnimationFrame(() => this.flush());
  }

  flush() {
    this.frameRequested = false;
    if (!this.pending) return;
    if (this.running) {
      this.request(this.reason);
      return;
    }
    const now = performance.now();
    const wait = this.minUpdateIntervalMs - (now - this.lastRun);
    if (wait > 0) {
      setTimeout(() => this.request(this.reason), wait);
      return;
    }
    this.pending = false;
    this.running = true;
    try {
      this.callback(this.reason);
      this.lastRun = performance.now();
    } finally {
      this.running = false;
      if (this.pending) this.request("latest");
    }
  }
}
