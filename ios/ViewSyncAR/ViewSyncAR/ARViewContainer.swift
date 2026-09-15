import ARKit
import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var model: AppModel

    func makeCoordinator() -> ViewSyncARCoordinator {
        ViewSyncARCoordinator(model: model)
    }

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        context.coordinator.attach(to: view)
        return view
    }

    func updateUIView(_ view: ARView, context: Context) {
        context.coordinator.setDebugMode(model.debugMode)
        context.coordinator.handle(command: model.command)
    }

    static func dismantleUIView(_ uiView: ARView, coordinator: ViewSyncARCoordinator) {
        coordinator.stop()
    }
}
