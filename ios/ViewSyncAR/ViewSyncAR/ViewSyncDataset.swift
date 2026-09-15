import Foundation
import RealityKit
import simd

@MainActor
struct ViewSyncDataset {
    let volume: MHAVolume
    let modelEntity: Entity
    let modelCenterMM: SIMD3<Float>
    let boundsMinimumMeters: SIMD3<Float>
    let boundsMaximumMeters: SIMD3<Float>

    var fieldOfViewMeters: Float {
        let size = boundsMaximumMeters - boundsMinimumMeters
        return max(size.x, max(size.y, size.z)) * 1.05
    }

    var surfaceToCenterDistance: Float {
        max(0.03, (boundsMaximumMeters.y - boundsMinimumMeters.y) / 2)
    }

    init(volume: MHAVolume, modelEntity: Entity, metadata: ViewSyncCoordinateMetadata) throws {
        self.volume = volume
        self.modelEntity = modelEntity
        modelCenterMM = try metadata.sourceCoordinateSystem.boundsCenter.simd3(name: "boundsCenter")
        boundsMinimumMeters = try metadata.trackingCoordinateSystem.boundsMinimum.simd3(name: "boundsMinimum")
        boundsMaximumMeters = try metadata.trackingCoordinateSystem.boundsMaximum.simd3(name: "boundsMaximum")
    }
}

struct ViewSyncCoordinateMetadata: Decodable, Sendable {
    struct SourceCoordinateSystem: Decodable, Sendable {
        let unit: String
        let upAxisAfterImport: String
        let boundsCenter: [Float]
    }

    struct TrackingCoordinateSystem: Decodable, Sendable {
        let unit: String
        let upAxis: String
        let boundsMinimum: [Float]
        let boundsMaximum: [Float]
    }

    struct SourceGLBToTracking: Decodable, Sendable {
        let scale: [Float]
        let translationMeters: [Float]
    }

    let dataset: String
    let sourceCoordinateSystem: SourceCoordinateSystem
    let trackingCoordinateSystem: TrackingCoordinateSystem
    let sourceGLBToTracking: SourceGLBToTracking

    func validate() throws {
        guard dataset == "260911",
              sourceCoordinateSystem.unit == "millimeter",
              sourceCoordinateSystem.upAxisAfterImport == "Y",
              trackingCoordinateSystem.unit == "meter",
              trackingCoordinateSystem.upAxis == "Y",
              sourceGLBToTracking.scale == [0.001, 0.001, 0.001],
              sourceGLBToTracking.translationMeters.count == 3 else {
            throw ViewSyncDatasetError.invalidMetadata("GLB→tracking座標契約が想定と一致しません")
        }
    }
}

enum ViewSyncDatasetError: LocalizedError {
    case invalidMetadata(String)

    var errorDescription: String? {
        switch self { case .invalidMetadata(let message): message }
    }
}

private extension Array where Element == Float {
    func simd3(name: String) throws -> SIMD3<Float> {
        guard count == 3 else {
            throw ViewSyncDatasetError.invalidMetadata("\(name)は3要素である必要があります")
        }
        return SIMD3(self[0], self[1], self[2])
    }
}

enum ViewSyncDatasetLoader {
    static func loadVolume(mhaURL: URL) throws -> MHAVolume {
        try MHALoader.load(url: mhaURL)
    }

    static func loadMetadata(url: URL) throws -> ViewSyncCoordinateMetadata {
        let metadata = try JSONDecoder().decode(
            ViewSyncCoordinateMetadata.self,
            from: Data(contentsOf: url, options: .mappedIfSafe)
        )
        try metadata.validate()
        return metadata
    }
}
