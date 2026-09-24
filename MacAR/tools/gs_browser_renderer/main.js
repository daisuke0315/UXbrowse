import {
    Application,
    Asset,
    AssetListLoader,
    Color,
    Entity,
    RESOLUTION_FIXED,
    Vec3,
    Texture, RenderTarget, PIXELFORMAT_RGBA32F, FILTER_NEAREST
} from './playcanvas.mjs';
import splatVertex from './gsplat-vert.js';
import splatFragment from './gsplat-frag.js';

const params = new URLSearchParams(window.location.search);
const numberParam = (name, fallback) => {
    const value = Number(params.get(name));
    return Number.isFinite(value) ? value : fallback;
};
const vecParam = (name, fallback) => {
    const text = params.get(name);
    if (!text) return fallback;
    const values = text.split(',').map(Number);
    return values.length === 3 && values.every(Number.isFinite) ? values : fallback;
};

const width = Math.max(1, Math.round(numberParam('width', 960)));
const height = Math.max(1, Math.round(numberParam('height', 720)));
const eye = vecParam('eye', [155.110230841, 831.726811996, -202.510051727]);
const target = vecParam('target', [8.328163147, 577.49281311, -202.510051727]);
const up = vecParam('up', [0, 0, 1]);
const fov = numberParam('fov', 55);
const canvas = document.getElementById('application-canvas');
const status = document.getElementById('status');
canvas.width = width;
canvas.height = height;

const app = new Application(canvas, {
    graphicsDeviceOptions: {
        antialias: false,
        alpha: false,
        powerPreference: 'high-performance'
    }
});
app.setCanvasResolution(RESOLUTION_FIXED, width, height);
app.autoRender = true;

const camera = new Entity('Camera');
camera.addComponent('camera', {
    clearColor: new Color(0.96, 0.96, 0.96, 1),
    fov,
    horizontalFov: false,
    nearClip: 0.01,
    farClip: 100000
});
camera.setPosition(eye[0], eye[1], eye[2]);
camera.lookAt(
    new Vec3(target[0], target[1], target[2]),
    new Vec3(up[0], up[1], up[2])
);
app.root.addChild(camera);

const applyCamera = (nextEye, nextTarget, nextUp, nextFov = fov) => {
    camera.setPosition(nextEye[0], nextEye[1], nextEye[2]);
    camera.lookAt(
        new Vec3(nextTarget[0], nextTarget[1], nextTarget[2]),
        new Vec3(nextUp[0], nextUp[1], nextUp[2])
    );
    camera.camera.fov = nextFov;
    app.renderNextFrame = true;
};

window.setCamera = (nextEye, nextTarget, nextUp, nextFov) => new Promise(resolve => {
    applyCamera(nextEye, nextTarget, nextUp, nextFov);
    let remaining = 8;
    const onFrameEnd = () => {
        remaining -= 1;
        if (remaining <= 0) {
            app.off('frameend', onFrameEnd);
            resolve(true);
        }
    };
    app.on('frameend', onFrameEnd);
});

const asset = new Asset('source-splat', 'gsplat', {
    url: './asset.ply',
    filename: 'asset.ply'
});
const loader = new AssetListLoader([asset], app.assets);

window.renderReady = false;
window.renderError = null;
status.textContent = 'loading PLY';
loader.load(() => {
    try {
        const splat = new Entity('Gaussian Splat');
        // Keep raw PLY coordinates. ViewSync camera poses and XYZ transfer buffers
        // are expressed in the exact same coordinate system.
        splat.addComponent('gsplat', { asset, unified: params.get('geometry') !== '1' });
        app.root.addChild(splat);
        if (params.get('geometry') === '1') {
            // Same splats, projected covariance, alpha cutoff and depth order as
            // appearance. Only the fragment payload changes. Not a surface mesh:
            // these are alpha-composited depth moments and occupancy.
            const material = splat.gsplat.material;
            const vertexAnchor = 'gaussianUV = corner.uv;';
            const fragmentAnchor = 'vec4 fragColor = vec4(gaussianColor.xyz, alpha);';
            if (!splatVertex.includes(vertexAnchor) || !splatFragment.includes(fragmentAnchor)) {
                throw new Error('Unsupported PlayCanvas shader; refusing mismatched geometry');
            }
            material.getShaderChunks().set('gsplatVS',
                'varying highp float geometryDepth;\n' + splatVertex.replace(vertexAnchor,
                    vertexAnchor + '\n geometryDepth = -center.view.z / 1000.0;'));
            material.getShaderChunks().set('gsplatPS',
                'varying highp float geometryDepth;\nuniform float geometryPass;\n' +
                splatFragment.replace(fragmentAnchor, fragmentAnchor + `
                    if (geometryPass > 0.5) {
                        fragColor.rgb = vec3(geometryDepth, geometryDepth * geometryDepth, 1.0);
                    }
                `));
            material.setParameter('geometryPass', 0);
            material.update();
            const texture = new Texture(app.graphicsDevice, {
                width, height, format: PIXELFORMAT_RGBA32F, mipmaps: false,
                minFilter: FILTER_NEAREST, magFilter: FILTER_NEAREST
            });
            const targetRT = new RenderTarget({ colorBuffer: texture, depth: true });
            window.captureGeometry = async () => {
                const device = app.graphicsDevice;
                if (!device.gl.getExtension('EXT_color_buffer_float') ||
                    !device.gl.getExtension('EXT_float_blend')) {
                    throw new Error('Float rendering/blending unavailable: cannot generate geometry');
                }
                material.setParameter('geometryPass', 1);
                camera.camera.clearColor = new Color(0, 0, 0, 0);
                camera.camera.renderTarget = targetRT;
                await new Promise(resolve => app.once('frameend', resolve));
                // Pause rendering during readback: no framebuffer race with RAF.
                app.autoRender = false;
                const pixels = new Float32Array(width * height * 4);
                device.setRenderTarget(targetRT);
                device.updateBegin();
                device.gl.readPixels(0, 0, width, height, device.gl.RGBA, device.gl.FLOAT, pixels);
                const error = device.gl.getError();
                device.updateEnd();
                material.setParameter('geometryPass', 0);
                camera.camera.renderTarget = null;
                camera.camera.clearColor = new Color(0.96, 0.96, 0.96, 1);
                app.autoRender = true;
                if (error) throw new Error(`Float readback error ${error}`);
                // WebGL readback is bottom-up. Store little-endian native floats
                // in top-down image order, matching the RGB screenshot.
                const flipped = new Float32Array(pixels.length);
                for (let y = 0; y < height; y++) {
                    flipped.set(pixels.subarray((height - 1 - y) * width * 4,
                        (height - y) * width * 4), y * width * 4);
                }
                const bytes = new Uint8Array(flipped.buffer);
                let binary = '';
                for (let offset = 0; offset < bytes.length; offset += 32768) {
                    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
                }
                return btoa(binary);
            };
        }
        applyCamera(eye, target, up, fov);
        let frames = 0;
        app.on('frameend', () => {
            frames += 1;
            status.textContent = `rendering frame ${frames}`;
            if (frames >= 8) {
                window.renderReady = true;
                status.remove();
            }
        });
        app.start();
    } catch (error) {
        window.renderError = String(error?.stack || error);
        status.textContent = window.renderError;
        console.error(error);
    }
});
asset.on('error', (error) => {
    window.renderError = String(error);
    status.textContent = window.renderError;
    console.error(error);
});
