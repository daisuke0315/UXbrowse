export class ValidationLogger {
  constructor(maxEntries = 300) {
    this.maxEntries = maxEntries;
    this.entries = [];
    this.listeners = new Set();
  }

  info(message, data) {
    this.add("info", message, data);
  }

  warn(message, data) {
    this.add("warn", message, data);
  }

  error(message, data) {
    this.add("error", message, data);
  }

  add(level, message, data) {
    const entry = {
      time: new Date().toLocaleTimeString(),
      level,
      message,
      data,
    };
    this.entries.push(entry);
    if (this.entries.length > this.maxEntries) this.entries.shift();
    for (const listener of this.listeners) listener(this.entries);
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.entries);
    return () => this.listeners.delete(listener);
  }

  toText() {
    return this.entries.map((entry) => {
      const suffix = entry.data ? ` ${JSON.stringify(entry.data)}` : "";
      return `${entry.time} [${entry.level.toUpperCase()}] ${entry.message}${suffix}`;
    }).join("\n");
  }
}
