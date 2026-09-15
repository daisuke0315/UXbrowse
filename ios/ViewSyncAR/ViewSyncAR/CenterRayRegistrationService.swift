import ARKit
import RealityKit
import simd

/// Estimates the model-space origin only while creating an ARReferenceObject.
/// This is enrollment calibration, never the runtime target-recognition result.
@MainActor
final class CenterRayRegistrationService {
    struct Result {
        let modelTransform: simd_float4x4
        let progress: Double
        let confidence: Double
        let distance: Float
    }

    private let requiredSamples = 10
    private let maximumSpread: Float = 0.028
    private var positions: [SIMD3<Float>] = []

    var progress: Double {
        min(1, Double(positions.count) / Double(requiredSamples))
    }

    func reset() {
        positions.removeAll(keepingCapacity: true)
    }

    func evaluate(
        in arView: ARView,
        frame: ARFrame,
        surfaceToCenterDistance: Float
    ) -> Result? {
        let point = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        guard let hit = arView.raycast(from: point, allowing: .estimatedPlane, alignment: .any).first else {
            if positions.count > 1 { positions.removeFirst() }
            return nil
        }

        let hitPosition = hit.worldTransform.translation
        if let mean = meanPosition(), simd_distance(mean, hitPosition) > 0.07 {
            positions.removeAll(keepingCapacity: true)
        }
        positions.append(hitPosition)
        if positions.count > requiredSamples { positions.removeFirst() }

        guard positions.count >= requiredSamples, let mean = meanPosition() else { return nil }
        let spread = positions.map { simd_distance($0, mean) }.max() ?? .greatestFiniteMagnitude
        guard spread <= maximumSpread else {
            positions.removeFirst(max(1, positions.count / 3))
            return nil
        }

        let camera = frame.camera.transform.translation
        let ray = simd_normalize(mean - camera)
        let modelCenter = mean + ray * surfaceToCenterDistance
        let distance = simd_distance(camera, modelCenter)
        guard distance > 0.12, distance < 2.0 else {
            reset()
            return nil
        }

        // ViewSync/DICOM model convention: Z is superior, Y is depth.
        // Keep model Z aligned with gravity and face model -Y toward the camera.
        var modelY = SIMD3(ray.x, 0, ray.z)
        if simd_length_squared(modelY) < 0.0001 { modelY = SIMD3(0, 0, -1) }
        modelY = simd_normalize(modelY)
        let modelZ = SIMD3<Float>(0, 1, 0)
        let modelX = simd_normalize(simd_cross(modelY, modelZ))
        var transform = matrix_identity_float4x4
        transform.columns.0 = SIMD4(modelX, 0)
        transform.columns.1 = SIMD4(modelY, 0)
        transform.columns.2 = SIMD4(modelZ, 0)
        transform.columns.3 = SIMD4(modelCenter, 1)
        let confidence = Double(max(0, min(1, 1 - spread / maximumSpread)))
        return Result(
            modelTransform: transform,
            progress: 1,
            confidence: confidence,
            distance: distance
        )
    }

    private func meanPosition() -> SIMD3<Float>? {
        guard !positions.isEmpty else { return nil }
        return positions.reduce(.zero, +) / Float(positions.count)
    }
}

extension simd_float4x4 {
    var translation: SIMD3<Float> {
        SIMD3(columns.3.x, columns.3.y, columns.3.z)
    }
}
