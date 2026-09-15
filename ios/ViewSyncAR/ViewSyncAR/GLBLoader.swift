import Foundation
import simd

struct ParsedGLB: @unchecked Sendable {
    let positionsMeters: [SIMD3<Float>]
    let textureCoordinates: [SIMD2<Float>]
    let indices: [UInt32]
    let texturePNG: Data?
    let modelCenterMM: SIMD3<Float>
    let boundsMinimumMeters: SIMD3<Float>
    let boundsMaximumMeters: SIMD3<Float>
}

enum GLBLoader {
    enum LoadError: LocalizedError {
        case invalid(String)

        var errorDescription: String? {
            switch self { case .invalid(let message): message }
        }
    }

    static func load(url: URL) throws -> ParsedGLB {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 28,
              data.readUInt32(at: 0) == 0x46546C67,
              data.readUInt32(at: 4) == 2 else {
            throw LoadError.invalid("GLB 2.0ファイルではありません")
        }
        let jsonLength = Int(data.readUInt32(at: 12))
        guard data.readUInt32(at: 16) == 0x4E4F534A,
              20 + jsonLength + 8 <= data.count else {
            throw LoadError.invalid("GLB JSON chunkが不正です")
        }
        let jsonData = data.subdata(in: 20..<(20 + jsonLength))
        guard let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let accessors = json["accessors"] as? [[String: Any]],
              let views = json["bufferViews"] as? [[String: Any]],
              let meshes = json["meshes"] as? [[String: Any]],
              let primitive = (meshes.first?["primitives"] as? [[String: Any]])?.first,
              let attributes = primitive["attributes"] as? [String: Any],
              let positionAccessor = attributes["POSITION"] as? Int,
              let uvAccessor = attributes["TEXCOORD_0"] as? Int,
              let indexAccessor = primitive["indices"] as? Int else {
            throw LoadError.invalid("GLB mesh構造を取得できません")
        }
        let binHeader = 20 + jsonLength
        guard data.readUInt32(at: binHeader + 4) == 0x004E4942 else {
            throw LoadError.invalid("GLB BIN chunkがありません")
        }
        let binOffset = binHeader + 8

        let positionInfo = try accessorInfo(accessors[positionAccessor], views: views, binOffset: binOffset)
        guard positionInfo.componentType == 5126, positionInfo.components == 3 else {
            throw LoadError.invalid("POSITIONはFloat32 VEC3である必要があります")
        }
        let minimum = try vector3(accessors[positionAccessor]["min"], name: "POSITION min")
        let maximum = try vector3(accessors[positionAccessor]["max"], name: "POSITION max")
        let center = (minimum + maximum) / 2
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(positionInfo.count)
        for index in 0..<positionInfo.count {
            let offset = positionInfo.offset + index * positionInfo.stride
            let point = SIMD3(
                data.readFloat32(at: offset),
                data.readFloat32(at: offset + 4),
                data.readFloat32(at: offset + 8)
            )
            positions.append((point - center) * 0.001)
        }

        let uvInfo = try accessorInfo(accessors[uvAccessor], views: views, binOffset: binOffset)
        guard uvInfo.componentType == 5126, uvInfo.components == 2,
              uvInfo.count == positions.count else {
            throw LoadError.invalid("TEXCOORD_0はPOSITIONと同数のFloat32 VEC2である必要があります")
        }
        var textureCoordinates: [SIMD2<Float>] = []
        textureCoordinates.reserveCapacity(uvInfo.count)
        for index in 0..<uvInfo.count {
            let offset = uvInfo.offset + index * uvInfo.stride
            textureCoordinates.append(SIMD2(data.readFloat32(at: offset), data.readFloat32(at: offset + 4)))
        }

        let indexInfo = try accessorInfo(accessors[indexAccessor], views: views, binOffset: binOffset)
        guard indexInfo.componentType == 5125, indexInfo.components == 1 else {
            throw LoadError.invalid("indicesはUInt32 SCALARである必要があります")
        }
        var indices: [UInt32] = []
        indices.reserveCapacity(indexInfo.count)
        for index in 0..<indexInfo.count {
            indices.append(data.readUInt32(at: indexInfo.offset + index * indexInfo.stride))
        }

        var texturePNG: Data?
        if let images = json["images"] as? [[String: Any]],
           let image = images.first,
           image["mimeType"] as? String == "image/png",
           let viewIndex = image["bufferView"] as? Int {
            let view = views[viewIndex]
            let offset = binOffset + (view["byteOffset"] as? Int ?? 0)
            let length = view["byteLength"] as? Int ?? 0
            if offset >= 0, length > 0, offset + length <= data.count {
                texturePNG = data.subdata(in: offset..<(offset + length))
            }
        }

        return ParsedGLB(
            positionsMeters: positions,
            textureCoordinates: textureCoordinates,
            indices: indices,
            texturePNG: texturePNG,
            modelCenterMM: center,
            boundsMinimumMeters: (minimum - center) * 0.001,
            boundsMaximumMeters: (maximum - center) * 0.001
        )
    }

    private struct AccessorInfo {
        let offset: Int
        let stride: Int
        let count: Int
        let componentType: Int
        let components: Int
    }

    private static func accessorInfo(
        _ accessor: [String: Any],
        views: [[String: Any]],
        binOffset: Int
    ) throws -> AccessorInfo {
        guard let viewIndex = accessor["bufferView"] as? Int,
              let count = accessor["count"] as? Int,
              let componentType = accessor["componentType"] as? Int,
              let type = accessor["type"] as? String else {
            throw LoadError.invalid("GLB accessorが不正です")
        }
        let components: Int
        switch type {
        case "SCALAR": components = 1
        case "VEC2": components = 2
        case "VEC3": components = 3
        case "VEC4": components = 4
        default: throw LoadError.invalid("未対応のaccessor type: \(type)")
        }
        let componentBytes: Int
        switch componentType {
        case 5121: componentBytes = 1
        case 5123: componentBytes = 2
        case 5125, 5126: componentBytes = 4
        default: throw LoadError.invalid("未対応のcomponent type: \(componentType)")
        }
        let view = views[viewIndex]
        let offset = binOffset + (view["byteOffset"] as? Int ?? 0) + (accessor["byteOffset"] as? Int ?? 0)
        let stride = view["byteStride"] as? Int ?? components * componentBytes
        return AccessorInfo(offset: offset, stride: stride, count: count, componentType: componentType, components: components)
    }

    private static func vector3(_ raw: Any?, name: String) throws -> SIMD3<Float> {
        guard let values = raw as? [NSNumber], values.count == 3 else {
            throw LoadError.invalid("GLBの\(name)が不正です")
        }
        return SIMD3(values[0].floatValue, values[1].floatValue, values[2].floatValue)
    }
}

private extension Data {
    func readUInt32(at offset: Int) -> UInt32 {
        withUnsafeBytes { raw in
            UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

    func readFloat32(at offset: Int) -> Float {
        Float(bitPattern: readUInt32(at: offset))
    }
}
