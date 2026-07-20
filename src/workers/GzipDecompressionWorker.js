self.addEventListener("message", async (event) => {
  try {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("このブラウザはgzip展開に対応していません");
    }
    const compressedBuffer = event.data?.buffer;
    if (!(compressedBuffer instanceof ArrayBuffer)) {
      throw new Error("圧縮CTデータが不正です");
    }
    const stream = new Blob([compressedBuffer]).stream().pipeThrough(new DecompressionStream("gzip"));
    const buffer = await new Response(stream).arrayBuffer();
    self.postMessage({ buffer }, [buffer]);
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
});
