import Foundation
import simd

@main
enum DatasetSmokeTest {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            throw TestError.usage
        }
        let volume = try MHALoader.load(url: URL(fileURLWithPath: CommandLine.arguments[1]))
        let model = try GLBLoader.load(url: URL(fileURLWithPath: CommandLine.arguments[2]))

        guard volume.dimensions == SIMD3(235, 223, 245),
              volume.voxels.count == 12_839_225,
              model.positionsMeters.count == 944_998,
              model.indices.count == 4_925_928,
              model.texturePNG != nil else {
            throw TestError.unexpectedDataset
        }
        guard volume.makeMPRImage(
            centerMM: model.modelCenterMM,
            u: SIMD3(1, 0, 0),
            v: SIMD3(0, 1, 0),
            fieldOfViewMM: 264,
            size: 64
        ) != nil else {
            throw TestError.mprFailed
        }

        print("260911 dataset smoke test passed")
        print("voxels: \(volume.voxels.count)")
        print("vertices: \(model.positionsMeters.count)")
        print("triangles: \(model.indices.count / 3)")
    }

    enum TestError: LocalizedError {
        case usage
        case unexpectedDataset
        case mprFailed

        var errorDescription: String? {
            switch self {
            case .usage: "usage: DatasetSmokeTest volume.mha model.glb"
            case .unexpectedDataset: "260911 dataset metadata did not match the expected values"
            case .mprFailed: "MPR image generation failed"
            }
        }
    }
}
