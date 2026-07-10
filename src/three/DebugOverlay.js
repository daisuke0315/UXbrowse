export class DebugOverlay {
  constructor(element) {
    this.element = element;
  }

  setText(text) {
    this.element.textContent = text;
  }
}
