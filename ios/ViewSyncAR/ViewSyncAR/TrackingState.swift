import SwiftUI

enum TrackingState: Equatable {
    case unsupported
    case needsReference
    case scanning
    case searching
    case locking
    case preview
    case tracking
    case lowConfidence
    case lost

    var title: String {
        switch self {
        case .unsupported: "AR非対応"
        case .needsReference: "TARGET NOT REGISTERED"
        case .scanning: "SCANNING TARGET"
        case .searching: "SEARCHING"
        case .locking: "LOCKING"
        case .preview: "LEGACY PREVIEW"
        case .tracking: "TRACKING LOCKED"
        case .lowConfidence: "LOW CONFIDENCE"
        case .lost: "TRACKING LOST"
        }
    }

    var instruction: String {
        switch self {
        case .unsupported: "この端末ではAR World Trackingを利用できません"
        case .needsReference: "中央を対象の正面中心へ合わせ、スキャンを開始してください"
        case .scanning: "対象の周囲をゆっくり移動して、全体を撮影してください"
        case .searching: "登録済み対象を画面内へ入れ、ゆっくり周囲を映してください"
        case .locking: "そのまま端末を静止してください"
        case .preview: "旧.arobject姿勢は検証用です。CT/MPRは安全のため停止しています"
        case .tracking: "端末を動かすとCT断面が追従します"
        case .lowConfidence: "ゆっくり動かし、対象物を画面内に保ってください"
        case .lost: "CT表示を停止しました。再登録してください"
        }
    }

    var tint: Color {
        switch self {
        case .tracking: .green
        case .locking, .scanning: .yellow
        case .preview, .lowConfidence: .orange
        case .lost, .unsupported: .red
        case .searching, .needsReference: .white
        }
    }

    var allowsMPR: Bool { self == .tracking }
}
