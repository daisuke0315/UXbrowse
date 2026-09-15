import CoreGraphics
import Foundation
import simd

struct MHAVolume: @unchecked Sendable {
    let voxels: [Float]
    let dimensions: SIMD3<Int>
    let spacing: SIMD3<Float>
    let origin: SIMD3<Float>
    let direction: simd_float3x3
    let inverseDirection: simd_float3x3

    func makeMPRImage(
        centerMM: SIMD3<Float>,
        u: SIMD3<Float>,
        v: SIMD3<Float>,
        fieldOfViewMM: Float,
        size: Int = 192,
        windowLevel: Float = 0,
        windowWidth: Float = 400
    ) -> CGImage? {
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        let pixelSpacing = fieldOfViewMM / Float(size)
        let half = Float(size - 1) / 2
        let minimum = windowLevel - windowWidth / 2

        for y in 0..<size {
            let offsetV = (half - Float(y)) * pixelSpacing
            for x in 0..<size {
                let offsetU = (Float(x) - half) * pixelSpacing
                let world = centerMM + u * offsetU + v * offsetV
                let sample = sampleLinear(atWorldMM: world)
                let index = (y * size + x) * 4
                guard let hu = sample else { continue }
                let value = max(0, min(1, (hu - minimum) / windowWidth))
                let gray = UInt8(clamping: Int((value * 255).rounded()))
                let alpha: UInt8 = hu <= -900 ? 0 : (hu < -500 ? 100 : 238)
                bytes[index] = gray
                bytes[index + 1] = gray
                bytes[index + 2] = gray
                bytes[index + 3] = alpha
            }
        }

        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: size,
            height: size,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    private func sampleLinear(atWorldMM point: SIMD3<Float>) -> Float? {
        let relative = point - origin
        let oriented = inverseDirection * relative
        let voxel = SIMD3(
            oriented.x / spacing.x,
            oriented.y / spacing.y,
            oriented.z / spacing.z
        )
        guard voxel.x >= 0, voxel.y >= 0, voxel.z >= 0,
              voxel.x <= Float(dimensions.x - 1),
              voxel.y <= Float(dimensions.y - 1),
              voxel.z <= Float(dimensions.z - 1) else { return nil }

        let x0 = Int(floor(voxel.x))
        let y0 = Int(floor(voxel.y))
        let z0 = Int(floor(voxel.z))
        let x1 = min(x0 + 1, dimensions.x - 1)
        let y1 = min(y0 + 1, dimensions.y - 1)
        let z1 = min(z0 + 1, dimensions.z - 1)
        let xd = voxel.x - Float(x0)
        let yd = voxel.y - Float(y0)
        let zd = voxel.z - Float(z0)

        let c00 = mix(value(x0, y0, z0), value(x1, y0, z0), xd)
        let c10 = mix(value(x0, y1, z0), value(x1, y1, z0), xd)
        let c01 = mix(value(x0, y0, z1), value(x1, y0, z1), xd)
        let c11 = mix(value(x0, y1, z1), value(x1, y1, z1), xd)
        return mix(mix(c00, c10, yd), mix(c01, c11, yd), zd)
    }

    private func value(_ x: Int, _ y: Int, _ z: Int) -> Float {
        voxels[z * dimensions.x * dimensions.y + y * dimensions.x + x]
    }

    private func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }
}

enum MHALoader {
    enum LoadError: LocalizedError {
        case invalid(String)

        var errorDescription: String? {
            switch self { case .invalid(let message): message }
        }
    }

    static func load(url: URL) throws -> MHAVolume {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let marker = "ElementDataFile = LOCAL".data(using: .ascii),
              let markerRange = data.range(of: marker),
              let newline = data[markerRange.upperBound...].firstIndex(of: 0x0A) else {
            throw LoadError.invalid("MHAのLOCALデータ開始位置を取得できません")
        }
        let dataOffset = newline + 1
        guard let header = String(data: data[..<dataOffset], encoding: .ascii) else {
            throw LoadError.invalid("MHAヘッダを解釈できません")
        }
        let fields = Dictionary(uniqueKeysWithValues: header.split(whereSeparator: \.isNewline).compactMap { line in
            let parts = line.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            return parts.count == 2 ? (parts[0].lowercased(), parts[1]) : nil
        })

        guard fields["elementtype"]?.uppercased() == "MET_FLOAT",
              fields["compresseddata"]?.lowercased() != "true" else {
            throw LoadError.invalid("現在は非圧縮MET_FLOAT MHAのみ対応しています")
        }
        let dimensions = try int3(fields["dimsize"], name: "DimSize")
        let spacing = try float3(fields["elementspacing"], name: "ElementSpacing")
        let origin = try float3(fields["offset"] ?? fields["origin"], name: "Offset")
        let matrixValues = try floats(fields["transformmatrix"], count: 9, name: "TransformMatrix")
        let direction = simd_float3x3(columns: (
            SIMD3(matrixValues[0], matrixValues[3], matrixValues[6]),
            SIMD3(matrixValues[1], matrixValues[4], matrixValues[7]),
            SIMD3(matrixValues[2], matrixValues[5], matrixValues[8])
        ))
        let voxelCount = dimensions.x * dimensions.y * dimensions.z
        let byteCount = voxelCount * MemoryLayout<Float>.size
        guard data.count - dataOffset >= byteCount else {
            throw LoadError.invalid("MHA voxel dataの長さが不足しています")
        }
        var voxels = [Float](repeating: 0, count: voxelCount)
        _ = voxels.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination, from: dataOffset..<(dataOffset + byteCount))
        }
        return MHAVolume(
            voxels: voxels,
            dimensions: dimensions,
            spacing: spacing,
            origin: origin,
            direction: direction,
            inverseDirection: direction.inverse
        )
    }

    private static func int3(_ raw: String?, name: String) throws -> SIMD3<Int> {
        let values = try floats(raw, count: 3, name: name).map(Int.init)
        guard values.allSatisfy({ $0 > 0 }) else { throw LoadError.invalid("\(name)が不正です") }
        return SIMD3(values[0], values[1], values[2])
    }

    private static func float3(_ raw: String?, name: String) throws -> SIMD3<Float> {
        let values = try floats(raw, count: 3, name: name)
        return SIMD3(values[0], values[1], values[2])
    }

    private static func floats(_ raw: String?, count: Int, name: String) throws -> [Float] {
        guard let raw else { throw LoadError.invalid("MHAに\(name)がありません") }
        let values = raw.split(whereSeparator: \.isWhitespace).compactMap { Float($0) }
        guard values.count == count else { throw LoadError.invalid("MHAの\(name)が不正です") }
        return values
    }
}
