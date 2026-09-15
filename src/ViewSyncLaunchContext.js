const SAFE_STUDY_KEY = /^[A-Za-z0-9._-]{1,128}$/;

export function resolveViewSyncLaunchContext(search, datasets, fallbackId) {
  const params = new URLSearchParams(search);
  const requestedDataset = params.get("dataset") || "";
  const dataset = datasets[requestedDataset] ?? datasets[fallbackId];
  if (!dataset) throw new Error(`ViewSync startup dataset is not configured: ${fallbackId}`);
  const requestedStudy = params.get("study") || "";
  return {
    dataset,
    studyKey: SAFE_STUDY_KEY.test(requestedStudy) ? requestedStudy : "",
    fromPspMock: params.get("source") === "psp-mock",
    usedFallback: Boolean(requestedDataset && !datasets[requestedDataset]),
  };
}
