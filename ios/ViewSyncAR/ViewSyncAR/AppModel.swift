import Foundation

@MainActor
final class AppModel: ObservableObject {
    struct Command: Equatable {
        enum Kind { case startEnrollment, finishEnrollment, resetDetection, deleteReference }
        let id = UUID()
        let kind: Kind
    }

    @Published var trackingState: TrackingState = .searching
    @Published var lockProgress: Double = 0
    @Published var statusDetail = "対象物へカメラを向けてください"
    @Published var metrics = TrackingMetrics.empty
    @Published var debugMode = false
    @Published var datasetReady = false
    @Published var datasetStatus = "260911データセットを読み込み中"
    @Published var referenceReady = false
    @Published var canDeleteReference = false
    @Published var diagnostics = RegistrationDiagnostics.empty
    @Published var diagnosticEvents: [DiagnosticEvent] = []
    @Published var diagnosticLogURL: URL?
    @Published private(set) var command: Command?

    func resetRegistration() {
        command = Command(kind: .resetDetection)
    }

    func startEnrollment() { command = Command(kind: .startEnrollment) }
    func finishEnrollment() { command = Command(kind: .finishEnrollment) }
    func deleteReference() { command = Command(kind: .deleteReference) }
}

struct RegistrationDiagnostics: Equatable {
    var session = "--------"
    var issueCode = "I000"
    var issueSummary = "diagnostics starting"
    var anchor = "not detected"
    var pose = "—"
    var screenOffset = "—"
    var rayBox = "not tested"
    var mpr = "blocked 0 / shown 0"

    static let empty = RegistrationDiagnostics()
}

struct DiagnosticEvent: Identifiable, Equatable {
    enum Level: String {
        case info = "I"
        case warning = "W"
        case error = "E"
    }

    let id = UUID()
    let elapsed: String
    let level: Level
    let code: String
    let message: String
}

struct TrackingMetrics: Equatable {
    var arFPS: Double
    var mprFPS: Double
    var registrationConfidence: Double
    var distanceMeters: Double?

    static let empty = TrackingMetrics(
        arFPS: 0,
        mprFPS: 0,
        registrationConfidence: 0,
        distanceMeters: nil
    )
}
