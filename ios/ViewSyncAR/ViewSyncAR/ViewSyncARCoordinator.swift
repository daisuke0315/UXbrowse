import ARKit
import Combine
import OSLog
import RealityKit
import UIKit
import simd

@MainActor
final class ViewSyncARCoordinator: NSObject, @preconcurrency ARSessionDelegate {
    private enum ReferenceSource: String {
        case none
        case legacyScan = "legacy-arobject"
        case trainedDigitalTwin = "trained-referenceobject"
    }

    private unowned let model: AppModel
    private weak var arView: ARView?
    private let enrollmentOriginEstimator = CenterRayRegistrationService()
    private var dataset: ViewSyncDataset?
    private var referenceObject: ARReferenceObject?
    private var referenceSource: ReferenceSource = .none

    private var registeredRoot: AnchorEntity?
    private var mprEntity: ModelEntity?
    private var mprTexture: TextureResource?
    private var detectedObjectIdentifier: UUID?
    private var registeredTransform: simd_float4x4?

    private var proposedEnrollmentTransform: simd_float4x4?
    private var enrollmentStartTime: TimeInterval?
    private var enrollmentCapturePending = false
    private var lastHandledCommandID: UUID?

    private var lastRegistrationTime: TimeInterval = 0
    private var lastMPRTime: TimeInterval = 0
    private var lastFrameTime: TimeInterval = 0
    private var lastNormalTrackingTime: TimeInterval = 0
    private var smoothedARFPS: Double = 0
    private var smoothedMPRFPS: Double = 0
    private var registrationConfidence: Double = 0
    private var currentDistance: Double?
    private var cameraTrackingIsNormal = false
    private var debugMode = false
    private var datasetLoadCancellable: AnyCancellable?

    private let diagnosticLogger = Logger(subsystem: "jp.viewsync.ViewSyncAR", category: "Registration")
    private var diagnostics = RegistrationDiagnostics.empty
    private var diagnosticStartUptime = ProcessInfo.processInfo.systemUptime
    private var diagnosticLogURL: URL?
    private var diagnosticEventTimes: [String: TimeInterval] = [:]
    private var lastDiagnosticPublishTime: TimeInterval = 0
    private var mprAttempts = 0
    private var mprBlocks = 0
    private var mprSuccesses = 0
    private var mprWasVisible = false

    init(model: AppModel) {
        self.model = model
    }

    func attach(to arView: ARView) {
        self.arView = arView
        prepareDiagnosticLog()
        recordDiagnostic(.info, code: "I001", message: "AR session attached", deduplicateFor: 0)
        arView.renderOptions.insert(.disableMotionBlur)
        arView.session.delegate = self
        arView.session.delegateQueue = .main

        guard ARWorldTrackingConfiguration.isSupported else {
            updateState(.unsupported)
            return
        }

        model.statusDetail = "260911データセットを読み込み中です"
        startRecognitionWhileDatasetLoads()
        loadBundledDataset()
    }

    func stop() {
        recordDiagnostic(.info, code: "I002", message: "AR session stopped", deduplicateFor: 0)
        arView?.session.pause()
    }

    func setDebugMode(_ enabled: Bool) {
        guard debugMode != enabled else { return }
        debugMode = enabled
        arView?.debugOptions = enabled ? [.showFeaturePoints, .showAnchorOrigins] : []
        recordDiagnostic(.info, code: "I003", message: "debug overlay \(enabled ? "ON" : "OFF")", deduplicateFor: 0)
    }

    func handle(command: AppModel.Command?) {
        guard let command, command.id != lastHandledCommandID else { return }
        lastHandledCommandID = command.id
        switch command.kind {
        case .startEnrollment:
            startEnrollment()
        case .finishEnrollment:
            finishEnrollment(force: true)
        case .resetDetection:
            restartDetection()
        case .deleteReference:
            deleteSavedReference()
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateARFPS(timestamp: frame.timestamp)
        handleCameraTracking(frame)

        switch model.trackingState {
        case .needsReference:
            updateEnrollmentOrigin(frame: frame)
        case .scanning:
            updateEnrollmentProgress(frame: frame)
        case .locking, .preview, .tracking, .lowConfidence:
            updateRegisteredTarget(frame: frame)
        default:
            break
        }
        publishMetrics(timestamp: frame.timestamp)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        guard let objectAnchor = anchors.compactMap({ $0 as? ARObjectAnchor }).first,
              objectAnchor.referenceObject.name == "260911" else { return }
        acceptDetectedObject(objectAnchor)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let identifier = detectedObjectIdentifier,
              let objectAnchor = anchors.compactMap({ $0 as? ARObjectAnchor }).first(where: { $0.identifier == identifier }) else { return }
        registeredTransform = objectAnchor.transform
        if let registeredTransform {
            diagnostics.pose = "\(millimeterVector(registeredTransform.translation)) mm"
        }
        applyRegisteredTransform()
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        guard let identifier = detectedObjectIdentifier,
              anchors.contains(where: { $0.identifier == identifier }) else { return }
        recordDiagnostic(.error, code: "E201", message: "OBJECT_ANCHOR_REMOVED", deduplicateFor: 0)
        failRegistration(message: "登録対象を見失いました")
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        recordDiagnostic(.error, code: "E202", message: "SESSION_FAILED \(error.localizedDescription)", deduplicateFor: 0)
        failRegistration(message: "ARセッションエラー: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        recordDiagnostic(.error, code: "E203", message: "SESSION_INTERRUPTED", deduplicateFor: 0)
        failRegistration(message: "ARセッションが中断されました")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        restartDetection()
    }

    private func loadBundledDataset() {
        guard let mhaURL = Bundle.main.url(forResource: "volume_preprocessed.model-bounds", withExtension: "mha"),
              let modelURL = Bundle.main.url(forResource: "260108-3-runtime", withExtension: "usdz"),
              let metadataURL = Bundle.main.url(forResource: "model-coordinate", withExtension: "json") else {
            recordDiagnostic(.error, code: "E101", message: "DATASET_RESOURCE_MISSING", deduplicateFor: 0)
            model.datasetStatus = "bundle resource missing"
            model.statusDetail = "260911データセットがアプリ内にありません"
            updateState(.lost, replaceDetail: false)
            return
        }

        let loadStart = ProcessInfo.processInfo.systemUptime
        recordDiagnostic(.info, code: "I100", message: "DATASET_LOAD_STARTED precompiled USDZ", deduplicateFor: 0)
        let dataTask = Task.detached(priority: .userInitiated) {
            let volume = try ViewSyncDatasetLoader.loadVolume(mhaURL: mhaURL)
            let metadata = try ViewSyncDatasetLoader.loadMetadata(url: metadataURL)
            return (volume, metadata)
        }

        datasetLoadCancellable = Entity.loadAsync(contentsOf: modelURL)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] completion in
                guard let self else { return }
                if case .failure(let error) = completion {
                    self.finishDatasetLoadFailure(error)
                }
            } receiveValue: { [weak self] modelEntity in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    do {
                        let (volume, metadata) = try await dataTask.value
                        self.dataset = try ViewSyncDataset(
                            volume: volume,
                            modelEntity: modelEntity,
                            metadata: metadata
                        )
                        self.model.datasetReady = true
                        let elapsed = ProcessInfo.processInfo.systemUptime - loadStart
                        self.model.datasetStatus = String(format: "260911 ready (%.2fs)", elapsed)
                        if let dataset = self.dataset {
                            let size = dataset.boundsMaximumMeters - dataset.boundsMinimumMeters
                            self.recordDiagnostic(
                                .info,
                                code: "I101",
                                message: String(format: "DATASET_READY %.2fs bbox=%@mm", elapsed, self.millimeterVector(size)),
                                deduplicateFor: 0
                            )
                        }
                        self.loadReferenceOrPrepareEnrollment()
                    } catch {
                        self.finishDatasetLoadFailure(error)
                    }
                }
            }
    }

    private func finishDatasetLoadFailure(_ error: Error) {
        recordDiagnostic(.error, code: "E102", message: "DATASET_LOAD_FAILED \(error.localizedDescription)", deduplicateFor: 0)
        model.datasetReady = false
        model.datasetStatus = "load failed"
        model.statusDetail = "データ読込失敗: \(error.localizedDescription)"
        updateState(.lost, replaceDetail: false)
    }

    private func loadReferenceOrPrepareEnrollment() {
        if referenceObject != nil {
            if let registeredTransform, registeredRoot == nil {
                installRegisteredObject(transform: registeredTransform)
                updateStateForReferenceSource()
            } else if registeredTransform == nil {
                updateState(.searching)
            }
            return
        }

        if let url = try? referenceArchiveURL(),
           FileManager.default.fileExists(atPath: url.path),
           let reference = try? ARReferenceObject(archiveURL: url) {
            referenceObject = reference
            referenceSource = .legacyScan
            model.referenceReady = true
            model.canDeleteReference = true
            recordDiagnostic(.info, code: "I110", message: "REFERENCE_LOADED 260911.arobject", deduplicateFor: 0)
            runWorldTracking(referenceObject: reference, reset: true)
            updateState(.searching)
        } else {
            referenceObject = nil
            model.referenceReady = false
            recordDiagnostic(.warning, code: "W110", message: "REFERENCE_NOT_FOUND enrollment required", deduplicateFor: 0)
            runObjectScanning(reset: true)
            updateState(.needsReference)
        }
    }

    private func startRecognitionWhileDatasetLoads() {
        if #available(iOS 27.0, *),
           let url = Bundle.main.url(forResource: "260911", withExtension: "referenceobject"),
           let reference = try? ARReferenceObject(archiveURL: url) {
            referenceObject = reference
            referenceSource = .trainedDigitalTwin
            model.referenceReady = true
            model.canDeleteReference = false
            recordDiagnostic(.info, code: "I108", message: "TRAINED_REFERENCE_LOADED 260911.referenceobject", deduplicateFor: 0)
            runWorldTracking(referenceObject: reference, reset: true)
            updateState(.searching)
        } else if let url = try? referenceArchiveURL(),
           FileManager.default.fileExists(atPath: url.path),
           let reference = try? ARReferenceObject(archiveURL: url) {
            referenceObject = reference
            referenceSource = .legacyScan
            model.referenceReady = true
            model.canDeleteReference = true
            recordDiagnostic(.info, code: "I109", message: "REFERENCE_EARLY_LOADED 260911.arobject", deduplicateFor: 0)
            runWorldTracking(referenceObject: reference, reset: true)
            updateState(.searching)
        } else {
            runWorldTracking(referenceObject: nil, reset: true)
        }
    }

    private func runWorldTracking(referenceObject: ARReferenceObject?, reset: Bool) {
        guard let arView else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        if let referenceObject { configuration.detectionObjects = [referenceObject] }
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        arView.session.run(configuration, options: options)
        recordDiagnostic(
            .info,
            code: "I120",
            message: "WORLD_TRACKING object=\(referenceObject == nil ? "OFF" : "ON") reset=\(reset)",
            deduplicateFor: 0
        )
    }

    private func runObjectScanning(reset: Bool) {
        guard let arView, ARObjectScanningConfiguration.isSupported else {
            model.statusDetail = "この端末では3D対象スキャンを利用できません"
            updateState(.unsupported, replaceDetail: false)
            return
        }
        let configuration = ARObjectScanningConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        arView.session.run(configuration, options: options)
        recordDiagnostic(.info, code: "I121", message: "OBJECT_SCANNING reset=\(reset)", deduplicateFor: 0)
        enrollmentOriginEstimator.reset()
        proposedEnrollmentTransform = nil
        enrollmentStartTime = nil
        enrollmentCapturePending = false
    }

    private func updateEnrollmentOrigin(frame: ARFrame) {
        guard let arView, let dataset,
              frame.timestamp - lastRegistrationTime >= 0.1 else { return }
        lastRegistrationTime = frame.timestamp
        if let result = enrollmentOriginEstimator.evaluate(
            in: arView,
            frame: frame,
            surfaceToCenterDistance: dataset.surfaceToCenterDistance
        ) {
            proposedEnrollmentTransform = result.modelTransform
            diagnostics.pose = "\(millimeterVector(result.modelTransform.translation)) mm"
            model.lockProgress = result.confidence
            model.statusDetail = "位置を取得しました。「スキャン開始」を押してください"
        } else {
            model.lockProgress = enrollmentOriginEstimator.progress
            model.statusDetail = TrackingState.needsReference.instruction
        }
    }

    private func startEnrollment() {
        guard model.trackingState == .needsReference else { return }
        guard proposedEnrollmentTransform != nil else {
            model.statusDetail = "正面中央へ合わせ、位置表示が安定するまで待ってください"
            return
        }
        enrollmentStartTime = arView?.session.currentFrame?.timestamp
        model.lockProgress = 0
        recordDiagnostic(.info, code: "I130", message: "ENROLLMENT_STARTED", deduplicateFor: 0)
        updateState(.scanning)
    }

    private func updateEnrollmentProgress(frame: ARFrame) {
        guard let start = enrollmentStartTime else { return }
        let elapsed = frame.timestamp - start
        model.lockProgress = min(1, elapsed / 10)
        model.statusDetail = elapsed < 4
            ? "対象の側面へゆっくり回り込んでください"
            : "上・下・反対側のテクスチャも撮影してください"
        if elapsed >= 12 { finishEnrollment(force: false) }
    }

    private func finishEnrollment(force: Bool) {
        guard model.trackingState == .scanning,
              !enrollmentCapturePending,
              let dataset,
              let transform = proposedEnrollmentTransform,
              let session = arView?.session else { return }
        let elapsed = (session.currentFrame?.timestamp ?? 0) - (enrollmentStartTime ?? 0)
        guard elapsed >= 4 || !force else {
            model.statusDetail = "最低4秒間、対象の周囲を撮影してください"
            return
        }
        enrollmentCapturePending = true
        updateState(.locking)
        model.statusDetail = "RGB特徴から対象登録データを生成しています"
        recordDiagnostic(
            .info,
            code: "I131",
            message: "REFERENCE_CAPTURE pose=\(millimeterVector(transform.translation))mm",
            deduplicateFor: 0
        )
        session.createReferenceObject(
            transform: transform,
            center: .zero,
            extent: dataset.boundsMaximumMeters - dataset.boundsMinimumMeters
        ) { [weak self] reference, error in
            Task { @MainActor in
                guard let self else { return }
                self.enrollmentCapturePending = false
                guard let reference else {
                    self.recordDiagnostic(
                        .error,
                        code: "E130",
                        message: "REFERENCE_CAPTURE_FAILED \(error?.localizedDescription ?? "insufficient features")",
                        deduplicateFor: 0
                    )
                    self.model.statusDetail = "対象登録失敗: \(error?.localizedDescription ?? "特徴点不足")"
                    self.updateState(.needsReference, replaceDetail: false)
                    return
                }
                reference.name = "260911"
                do {
                    let url = try self.referenceArchiveURL()
                    try reference.export(to: url, previewImage: nil)
                    self.referenceObject = reference
                    self.referenceSource = .legacyScan
                    self.model.referenceReady = true
                    self.model.canDeleteReference = true
                    self.recordDiagnostic(.info, code: "I132", message: "REFERENCE_SAVED 260911.arobject", deduplicateFor: 0)
                    self.runWorldTracking(referenceObject: reference, reset: true)
                    self.updateState(.searching)
                } catch {
                    self.recordDiagnostic(.error, code: "E131", message: "REFERENCE_SAVE_FAILED \(error.localizedDescription)", deduplicateFor: 0)
                    self.model.statusDetail = "対象登録の保存失敗: \(error.localizedDescription)"
                    self.updateState(.needsReference, replaceDetail: false)
                }
            }
        }
    }

    private func referenceArchiveURL() throws -> URL {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw CocoaError(.fileNoSuchFile)
        }
        let directory = base.appendingPathComponent("ViewSyncAR", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("260911.arobject")
    }

    private func deleteSavedReference() {
        guard referenceSource != .trainedDigitalTwin else {
            restartDetection()
            return
        }
        if let url = try? referenceArchiveURL(), FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
        referenceObject = nil
        referenceSource = .none
        model.referenceReady = false
        model.canDeleteReference = false
        recordDiagnostic(.warning, code: "W111", message: "REFERENCE_DELETED", deduplicateFor: 0)
        removeRegisteredContent()
        runObjectScanning(reset: true)
        updateState(.needsReference)
    }

    private func restartDetection() {
        removeRegisteredContent()
        registrationConfidence = 0
        model.lockProgress = 0
        mprWasVisible = false
        recordDiagnostic(.info, code: "I200", message: "RECOGNITION_RESTART", deduplicateFor: 0)
        if let referenceObject {
            runWorldTracking(referenceObject: referenceObject, reset: true)
            updateState(.searching)
        } else if model.datasetReady {
            runObjectScanning(reset: true)
            updateState(.needsReference)
        }
    }

    private func acceptDetectedObject(_ objectAnchor: ARObjectAnchor) {
        removeRegisteredContent()
        detectedObjectIdentifier = objectAnchor.identifier
        registeredTransform = objectAnchor.transform
        registrationConfidence = referenceSource == .trainedDigitalTwin ? 1 : 0.35
        let anchorID = String(objectAnchor.identifier.uuidString.prefix(8)).uppercased()
        diagnostics.anchor = "detected #\(anchorID)"
        diagnostics.pose = "\(millimeterVector(objectAnchor.transform.translation)) mm"
        diagnostics.screenOffset = "pending projection"
        diagnostics.rayBox = "not tested"
        recordDiagnostic(
            .info,
            code: "I210",
            message: "OBJECT_DETECTED #\(anchorID) source=\(referenceSource.rawValue) pose=\(millimeterVector(objectAnchor.transform.translation))mm",
            deduplicateFor: 0
        )
        if dataset != nil {
            installRegisteredObject(transform: objectAnchor.transform)
        } else {
            recordDiagnostic(.info, code: "I211", message: "OBJECT_DETECTED_DATASET_PENDING", deduplicateFor: 0)
        }
        updateStateForReferenceSource()
    }

    private func updateRegisteredTarget(frame: ARFrame) {
        guard registeredRoot != nil else { return }
        if let transform = registeredTransform {
            currentDistance = Double(simd_distance(frame.camera.transform.translation, transform.translation))
        }

        guard referenceSource == .trainedDigitalTwin else {
            registrationConfidence = 0.35
            model.lockProgress = registrationConfidence
            markMPRInactive()
            updateState(.preview)
            model.statusDetail = "旧.arobject姿勢のモデル表示のみ。referenceobject検証完了までMPRは停止します"
            recordDiagnostic(
                .warning,
                code: "W260",
                message: "LEGACY_POSE_PREVIEW_ONLY MPR_BLOCKED referenceobject required",
                deduplicateFor: 5
            )
            return
        }

        guard cameraTrackingIsNormal else {
            registrationConfidence = 0
            model.lockProgress = 0
            markMPRInactive()
            return
        }

        registrationConfidence = 1
        model.lockProgress = 1
        updateState(.tracking)
        model.statusDetail = "学習済みdigital twinをARKitで追跡中"

        if model.trackingState == .tracking,
           frame.timestamp - lastMPRTime >= 1.0 / 20.0 {
            updateMPR(frame: frame)
        }
    }

    private func updateStateForReferenceSource() {
        guard dataset != nil else {
            updateState(.locking)
            model.statusDetail = "対象を認識しました。モデルとCTの読込完了を待っています"
            return
        }
        if referenceSource == .trainedDigitalTwin {
            if cameraTrackingIsNormal {
                updateState(.tracking)
                model.statusDetail = "学習済みdigital twinをARKitで追跡中"
            } else {
                updateState(.locking)
                model.statusDetail = "ARKitのカメラ姿勢を初期化しています"
            }
        } else {
            updateState(.preview)
            model.statusDetail = "旧.arobject姿勢のモデル表示のみ。referenceobject検証完了までMPRは停止します"
        }
    }

    private func installRegisteredObject(transform: simd_float4x4) {
        guard let arView, let dataset else { return }
        let root = AnchorEntity(world: transform)
        recordDiagnostic(.info, code: "I230", message: "AR_CONTENT_INSTALLED", deduplicateFor: 0)

        let modelOverlay = dataset.modelEntity.clone(recursive: true)
        makeModelOverlayTransparent(modelOverlay)
        root.addChild(modelOverlay)

        let initialImage = dataset.volume.makeMPRImage(
            centerMM: dataset.modelCenterMM,
            u: SIMD3(1, 0, 0),
            v: SIMD3(0, 0, 1),
            fieldOfViewMM: dataset.fieldOfViewMeters * 1_000
        )
        var mprMaterial = UnlitMaterial()
        if let initialImage,
           let texture = try? TextureResource.generate(
               from: initialImage,
               withName: "260911 CT MPR",
               options: .init(semantic: .color, mipmapsMode: .none)
           ) {
            mprTexture = texture
            mprMaterial.color = .init(tint: .white, texture: .init(texture))
        } else {
            mprMaterial.color = .init(tint: .white.withAlphaComponent(0.85))
            recordDiagnostic(.error, code: "E230", message: "INITIAL_MPR_TEXTURE_FAILED", deduplicateFor: 0)
        }
        mprMaterial.blending = .transparent(opacity: 0.95)
        let mpr = ModelEntity(
            mesh: .generatePlane(width: dataset.fieldOfViewMeters, height: dataset.fieldOfViewMeters),
            materials: [mprMaterial]
        )
        mpr.isEnabled = false
        root.addChild(mpr)
        arView.scene.addAnchor(root)
        registeredRoot = root
        mprEntity = mpr
    }

    private func makeModelOverlayTransparent(_ entity: Entity) {
        if let modelEntity = entity as? ModelEntity, var component = modelEntity.model {
            component.materials = component.materials.map { material in
                if var physicallyBased = material as? PhysicallyBasedMaterial {
                    physicallyBased.blending = .transparent(opacity: 0.34)
                    return physicallyBased
                }
                if var unlit = material as? UnlitMaterial {
                    unlit.blending = .transparent(opacity: 0.34)
                    return unlit
                }
                return material
            }
            modelEntity.model = component
        }
        for child in entity.children {
            makeModelOverlayTransparent(child)
        }
    }

    private func applyRegisteredTransform() {
        guard let transform = registeredTransform else { return }
        registeredRoot?.setTransformMatrix(transform, relativeTo: nil)
    }

    private func updateMPR(frame: ARFrame) {
        guard let root = registeredRoot, let mprEntity, let dataset else { return }
        mprAttempts += 1
        let updateInterval = lastMPRTime > 0 ? frame.timestamp - lastMPRTime : 0
        lastMPRTime = frame.timestamp
        let cameraPosition = frame.camera.transform.translation
        let cameraForward = -SIMD3<Float>(
            frame.camera.transform.columns.2.x,
            frame.camera.transform.columns.2.y,
            frame.camera.transform.columns.2.z
        )
        let modelTransform = root.transformMatrix(relativeTo: nil)
        let worldToModel = modelTransform.inverse
        let localCamera4 = worldToModel * SIMD4(cameraPosition, 1)
        let localDirection4 = worldToModel * SIMD4(cameraForward, 0)
        let localCamera = SIMD3(localCamera4.x, localCamera4.y, localCamera4.z)
        let localDirection = simd_normalize(SIMD3(localDirection4.x, localDirection4.y, localDirection4.z))
        guard let interval = rayBoxInterval(
            origin: localCamera,
            direction: localDirection,
            minimum: dataset.boundsMinimumMeters,
            maximum: dataset.boundsMaximumMeters
        ) else {
            mprBlocks += 1
            markMPRInactive()
            diagnostics.rayBox = "MPR MISS o[\(compactVector(localCamera))] d[\(compactVector(localDirection))]"
            diagnostics.mpr = "BLOCK \(mprBlocks) / shown \(mprSuccesses)"
            recordDiagnostic(
                .warning,
                code: "W220",
                message: "MPR_RAY_MISS offset=\(diagnostics.screenOffset) o[\(compactVector(localCamera))] d[\(compactVector(localDirection))]",
                deduplicateFor: 1
            )
            model.statusDetail = "中央の円を登録対象へ戻してください"
            return
        }
        diagnostics.rayBox = String(
            format: "MPR HIT near %.0f far %.0fmm",
            interval.near * 1_000,
            interval.far * 1_000
        )

        let localCenter = localCamera + localDirection * ((max(interval.near, 0) + interval.far) / 2)
        let center4 = modelTransform * SIMD4(localCenter, 1)
        let center = SIMD3(center4.x, center4.y, center4.z)
        let n = simd_normalize(center - cameraPosition)
        let cameraUp = SIMD3<Float>(
            frame.camera.transform.columns.1.x,
            frame.camera.transform.columns.1.y,
            frame.camera.transform.columns.1.z
        )
        var u = simd_cross(cameraUp, n)
        if simd_length_squared(u) < 0.0001 { u = SIMD3(1, 0, 0) }
        u = simd_normalize(u)
        let v = simd_normalize(simd_cross(n, u))

        var planeTransform = matrix_identity_float4x4
        planeTransform.columns.0 = SIMD4(u, 0)
        planeTransform.columns.1 = SIMD4(v, 0)
        planeTransform.columns.2 = SIMD4(n, 0)
        planeTransform.columns.3 = SIMD4(center, 1)
        mprEntity.setTransformMatrix(planeTransform, relativeTo: nil)

        let localU4 = worldToModel * SIMD4(u, 0)
        let localV4 = worldToModel * SIMD4(v, 0)
        let localU = simd_normalize(SIMD3(localU4.x, localU4.y, localU4.z))
        let localV = simd_normalize(SIMD3(localV4.x, localV4.y, localV4.z))
        let centerModelMM = localCenter * 1_000 + dataset.modelCenterMM
        guard let image = dataset.volume.makeMPRImage(
            centerMM: centerModelMM,
            u: localU,
            v: localV,
            fieldOfViewMM: dataset.fieldOfViewMeters * 1_000
        ) else {
            mprBlocks += 1
            markMPRInactive()
            diagnostics.mpr = "MHA_SAMPLE_FAILED"
            recordDiagnostic(
                .error,
                code: "E221",
                message: "MPR_VOLUME_SAMPLE_FAILED center=\(millimeterVector(centerModelMM * 0.001))mm",
                deduplicateFor: 1
            )
            return
        }
        guard let mprTexture else {
            mprBlocks += 1
            markMPRInactive()
            diagnostics.mpr = "TEXTURE_MISSING"
            recordDiagnostic(.error, code: "E222", message: "MPR_TEXTURE_MISSING", deduplicateFor: 1)
            return
        }
        do {
            try mprTexture.replace(
                withImage: image,
                options: .init(semantic: .color, mipmapsMode: .none)
            )
        } catch {
            mprBlocks += 1
            markMPRInactive()
            diagnostics.mpr = "TEXTURE_UPDATE_FAILED"
            recordDiagnostic(.error, code: "E223", message: "MPR_TEXTURE_UPDATE_FAILED \(error.localizedDescription)", deduplicateFor: 1)
            return
        }
        mprEntity.isEnabled = true
        mprSuccesses += 1
        diagnostics.mpr = "VISIBLE \(mprSuccesses) / blocked \(mprBlocks)"
        if !mprWasVisible {
            recordDiagnostic(
                .info,
                code: "I300",
                message: "MPR_VISIBLE center=\(millimeterVector(centerModelMM * 0.001))mm offset=\(diagnostics.screenOffset)",
                deduplicateFor: 0
            )
        }
        mprWasVisible = true
        model.statusDetail = TrackingState.tracking.instruction
        if updateInterval > 0 {
            let instantaneous = 1.0 / updateInterval
            smoothedMPRFPS = smoothedMPRFPS == 0 ? instantaneous : smoothedMPRFPS * 0.85 + instantaneous * 0.15
        }
    }

    private func rayBoxInterval(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        minimum: SIMD3<Float>,
        maximum: SIMD3<Float>
    ) -> (near: Float, far: Float)? {
        var near: Float = -.greatestFiniteMagnitude
        var far: Float = .greatestFiniteMagnitude
        for axis in 0..<3 {
            if abs(direction[axis]) < 0.000_001 {
                if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] { return nil }
                continue
            }
            let first = (minimum[axis] - origin[axis]) / direction[axis]
            let second = (maximum[axis] - origin[axis]) / direction[axis]
            near = max(near, min(first, second))
            far = min(far, max(first, second))
            if near > far { return nil }
        }
        guard far >= max(near, 0) else { return nil }
        return (near, far)
    }

    private func handleCameraTracking(_ frame: ARFrame) {
        switch frame.camera.trackingState {
        case .normal:
            cameraTrackingIsNormal = true
            lastNormalTrackingTime = frame.timestamp
        case .limited(let reason):
            cameraTrackingIsNormal = false
            markMPRInactive()
            guard registeredRoot != nil else { return }
            let elapsed = frame.timestamp - lastNormalTrackingTime
            if elapsed > 1.25 {
                recordDiagnostic(.error, code: "E240", message: "CAMERA_TRACKING_LOST \(trackingMessage(for: reason))", deduplicateFor: 1)
                failRegistration(message: trackingMessage(for: reason))
            } else if elapsed > 0.35 {
                markMPRInactive()
                recordDiagnostic(.warning, code: "W240", message: "CAMERA_TRACKING_LIMITED \(trackingMessage(for: reason))", deduplicateFor: 1)
                model.statusDetail = trackingMessage(for: reason)
                updateState(.lowConfidence, replaceDetail: false)
            }
        case .notAvailable:
            cameraTrackingIsNormal = false
            markMPRInactive()
            guard registeredRoot != nil else { return }
            failRegistration(message: "カメラ位置を取得できません")
        }
    }

    private func trackingMessage(for reason: ARCamera.TrackingState.Reason) -> String {
        switch reason {
        case .excessiveMotion: "端末の動きが速すぎます"
        case .insufficientFeatures: "周囲のテクスチャ特徴が不足しています"
        case .initializing: "AR位置を再推定しています"
        case .relocalizing: "AR空間へ再ローカライズしています"
        @unknown default: "Tracking confidenceが低下しました"
        }
    }

    private func failRegistration(message: String) {
        markMPRInactive()
        model.statusDetail = message
        updateState(.lost, replaceDetail: false)
    }

    private func removeRegisteredContent() {
        markMPRInactive()
        if let registeredRoot { arView?.scene.removeAnchor(registeredRoot) }
        registeredRoot = nil
        mprEntity = nil
        mprTexture = nil
        detectedObjectIdentifier = nil
        registeredTransform = nil
        diagnostics.anchor = "not detected"
        diagnostics.pose = "—"
        diagnostics.screenOffset = "—"
        diagnostics.rayBox = "not tested"
        diagnostics.mpr = "blocked \(mprBlocks) / shown \(mprSuccesses)"
        mprWasVisible = false
        currentDistance = nil
    }

    private func markMPRInactive() {
        mprEntity?.isEnabled = false
        mprWasVisible = false
        smoothedMPRFPS = 0
    }

    private func updateARFPS(timestamp: TimeInterval) {
        defer { lastFrameTime = timestamp }
        guard lastFrameTime > 0 else { return }
        let instantaneous = 1.0 / max(timestamp - lastFrameTime, 1.0 / 120.0)
        smoothedARFPS = smoothedARFPS == 0 ? instantaneous : smoothedARFPS * 0.9 + instantaneous * 0.1
    }

    private func publishMetrics(timestamp: TimeInterval) {
        model.metrics = TrackingMetrics(
            arFPS: smoothedARFPS,
            mprFPS: smoothedMPRFPS,
            registrationConfidence: registrationConfidence,
            distanceMeters: currentDistance
        )
        if timestamp - lastDiagnosticPublishTime >= 0.2 {
            lastDiagnosticPublishTime = timestamp
            model.diagnostics = diagnostics
        }
    }

    private func updateState(_ state: TrackingState, replaceDetail: Bool = true) {
        guard model.trackingState != state else { return }
        let previous = model.trackingState
        model.trackingState = state
        if replaceDetail { model.statusDetail = state.instruction }
        recordDiagnostic(
            .info,
            code: "I010",
            message: "STATE \(previous.title) -> \(state.title)",
            deduplicateFor: 0
        )
    }

    private func prepareDiagnosticLog() {
        diagnosticStartUptime = ProcessInfo.processInfo.systemUptime
        let session = String(UUID().uuidString.prefix(8)).uppercased()
        diagnostics = .empty
        diagnostics.session = session
        model.diagnostics = diagnostics
        model.diagnosticEvents = []
        diagnosticEventTimes.removeAll(keepingCapacity: true)

        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return
        }
        let directory = base.appendingPathComponent("ViewSyncAR", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent("registration-\(session).log")
            let header = "ViewSync AR registration trace #\(session)\n"
            try Data(header.utf8).write(to: url, options: .atomic)
            diagnosticLogURL = url
            model.diagnosticLogURL = url
        } catch {
            diagnosticLogURL = nil
            model.diagnosticLogURL = nil
            diagnosticLogger.error("Unable to create diagnostic log: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func recordDiagnostic(
        _ level: DiagnosticEvent.Level,
        code: String,
        message: String,
        deduplicateFor interval: TimeInterval = 1
    ) {
        diagnostics.issueCode = code
        diagnostics.issueSummary = message

        let uptime = ProcessInfo.processInfo.systemUptime
        if let previous = diagnosticEventTimes[code], uptime - previous < interval {
            return
        }
        diagnosticEventTimes[code] = uptime
        let elapsed = max(0, uptime - diagnosticStartUptime)
        let elapsedText = String(format: "+%06.2fs", elapsed)
        let event = DiagnosticEvent(
            elapsed: elapsedText,
            level: level,
            code: code,
            message: message
        )
        model.diagnosticEvents.insert(event, at: 0)
        if model.diagnosticEvents.count > 40 {
            model.diagnosticEvents.removeLast(model.diagnosticEvents.count - 40)
        }
        model.diagnostics = diagnostics

        let line = "\(elapsedText) [\(code)] \(message)"
        switch level {
        case .info:
            diagnosticLogger.info("\(line, privacy: .public)")
        case .warning:
            diagnosticLogger.warning("\(line, privacy: .public)")
        case .error:
            diagnosticLogger.error("\(line, privacy: .public)")
        }
        appendDiagnosticLine(line)
    }

    private func appendDiagnosticLine(_ line: String) {
        guard let diagnosticLogURL,
              let data = "\(line)\n".data(using: .utf8),
              let handle = try? FileHandle(forWritingTo: diagnosticLogURL) else { return }
        defer { try? handle.close() }
        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } catch {
            diagnosticLogger.error("Unable to append diagnostic log: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func millimeterVector(_ value: SIMD3<Float>) -> String {
        String(format: "%.0f,%.0f,%.0f", value.x * 1_000, value.y * 1_000, value.z * 1_000)
    }

    private func compactVector(_ value: SIMD3<Float>) -> String {
        String(format: "%.2f,%.2f,%.2f", value.x, value.y, value.z)
    }
}
