import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ZStack {
            ARViewContainer(model: model)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                statusCard
                    .padding(.horizontal, 16)
                    .padding(.top, 10)

                Spacer()

                crosshair

                Spacer()

                if model.debugMode {
                    debugCard
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)
                }

                controls
                    .padding(.horizontal, 16)
                    .padding(.bottom, 10)
            }
        }
    }

    private var statusCard: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(model.trackingState.tint)
                .frame(width: 10, height: 10)
                .shadow(color: model.trackingState.tint, radius: 5)

            VStack(alignment: .leading, spacing: 3) {
                Text(model.trackingState.title)
                    .font(.system(.headline, design: .rounded, weight: .bold))
                Text(model.statusDetail.isEmpty ? model.trackingState.instruction : model.statusDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            if model.trackingState == .locking || model.trackingState == .scanning {
                ProgressView(value: model.lockProgress)
                    .progressViewStyle(.circular)
                    .tint(.yellow)
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .overlay {
            RoundedRectangle(cornerRadius: 18)
                .stroke(model.trackingState.tint.opacity(0.45), lineWidth: 1)
        }
    }

    private var crosshair: some View {
        ZStack {
            Circle()
                .stroke(model.trackingState.tint.opacity(0.9), lineWidth: 2)
                .frame(width: 58, height: 58)
            Rectangle()
                .fill(model.trackingState.tint)
                .frame(width: 18, height: 1)
            Rectangle()
                .fill(model.trackingState.tint)
                .frame(width: 1, height: 18)
        }
        .shadow(color: .black.opacity(0.6), radius: 2)
        .animation(.easeInOut(duration: 0.2), value: model.trackingState)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                performPrimaryAction()
            } label: {
                Label(primaryButtonTitle, systemImage: primaryButtonIcon)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(ActionButtonStyle())
            .disabled(!model.datasetReady)

            if model.referenceReady && model.canDeleteReference && model.trackingState != .scanning {
                Button {
                    model.deleteReference()
                } label: {
                    Image(systemName: "trash")
                        .frame(width: 24)
                }
                .buttonStyle(ActionButtonStyle())
                .accessibilityLabel("対象登録を削除")
            }

            Button {
                model.debugMode.toggle()
            } label: {
                Image(systemName: model.debugMode ? "waveform.path.ecg.rectangle.fill" : "waveform.path.ecg.rectangle")
                    .frame(width: 24)
            }
            .buttonStyle(ActionButtonStyle())
            .accessibilityLabel("デバッグ表示")
        }
    }

    private var primaryButtonTitle: String {
        switch model.trackingState {
        case .needsReference: "スキャン開始"
        case .scanning: "スキャン完了"
        default: "再認識"
        }
    }

    private var primaryButtonIcon: String {
        switch model.trackingState {
        case .needsReference: "viewfinder"
        case .scanning: "checkmark.circle"
        default: "scope"
        }
    }

    private func performPrimaryAction() {
        switch model.trackingState {
        case .needsReference: model.startEnrollment()
        case .scanning: model.finishEnrollment()
        default: model.resetRegistration()
        }
    }

    private var debugCard: some View {
        let distance = model.metrics.distanceMeters.map { String(format: "%.0f mm", $0 * 1_000) } ?? "—"
        return VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("REGISTRATION TRACE  #\(model.diagnostics.session)")
                    .font(.system(.caption2, design: .monospaced, weight: .bold))
                    .foregroundStyle(.cyan)
                Spacer()
                if let logURL = model.diagnosticLogURL {
                    ShareLink(item: logURL) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.caption)
                            .foregroundStyle(.cyan)
                    }
                    .accessibilityLabel("診断ログを共有")
                }
            }
            Text("\(model.diagnostics.issueCode)  \(model.diagnostics.issueSummary)")
                .font(.system(.caption, design: .monospaced, weight: .bold))
                .foregroundStyle(issueColor(model.diagnostics.issueCode))
                .lineLimit(2)
            metric("AR / MPR", String(format: "%.1f / %.1f fps", model.metrics.arFPS, model.metrics.mprFPS))
            metric("confidence", String(format: "%.0f%%", model.metrics.registrationConfidence * 100))
            metric("target distance", distance)
            metric("dataset", model.datasetStatus)
            metric("object anchor", model.diagnostics.anchor)
            metric("pose xyz", model.diagnostics.pose)
            metric("center offset", model.diagnostics.screenOffset)
            metric("ray → bbox", model.diagnostics.rayBox)
            metric("MPR", model.diagnostics.mpr)

            Divider().overlay(.white.opacity(0.25))
            ForEach(model.diagnosticEvents.prefix(4)) { event in
                Text("\(event.elapsed) \(event.code) \(event.message)")
                    .foregroundStyle(eventColor(event.level))
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
            }
        }
        .font(.system(.caption2, design: .monospaced))
        .padding(12)
        .background(.black.opacity(0.68), in: RoundedRectangle(cornerRadius: 12))
    }

    private func metric(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name).foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
    }

    private func issueColor(_ code: String) -> Color {
        if code.hasPrefix("E") { return .red }
        if code.hasPrefix("W") { return .yellow }
        return .green
    }

    private func eventColor(_ level: DiagnosticEvent.Level) -> Color {
        switch level {
        case .info: .white.opacity(0.82)
        case .warning: .yellow
        case .error: .red
        }
    }
}

private struct ActionButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .padding(.vertical, 13)
            .padding(.horizontal, 18)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .opacity(configuration.isPressed ? 0.65 : 1)
    }
}
