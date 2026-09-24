import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from '../../.tools/gs-browser/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(scriptDir, '../..');
const playcanvasPath = path.join(
    projectDir,
    '.tools/gs-browser/node_modules/playcanvas/build/playcanvas.mjs'
);
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [planPathText, outputPathText] = process.argv.slice(2);

if (!planPathText || !outputPathText) {
    console.error('Usage: node capture_batch.mjs PLAN.json OUTPUT_DIR');
    process.exit(2);
}

const planPath = path.resolve(planPathText);
const outputDir = path.resolve(outputPathText);
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const staticFiles = new Map([
    ['/', [path.join(scriptDir, 'index.html'), 'text/html; charset=utf-8']],
    ['/index.html', [path.join(scriptDir, 'index.html'), 'text/html; charset=utf-8']],
    ['/main.js', [path.join(scriptDir, 'main.js'), 'text/javascript; charset=utf-8']],
    ['/playcanvas.mjs', [playcanvasPath, 'text/javascript; charset=utf-8']],
    ['/asset.ply', [path.resolve(plan.ply), 'application/octet-stream']]
]);
for (const stage of ['vert', 'frag']) {
    staticFiles.set(`/gsplat-${stage}.js`, [path.join(projectDir,
        `.tools/gs-browser/node_modules/playcanvas/build/playcanvas/src/scene/shader-lib/glsl/chunks/gsplat/${stage}/gsplat.js`),
        'text/javascript; charset=utf-8']);
}

const server = http.createServer((request, response) => {
    const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
    if (pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
    }
    const entry = staticFiles.get(pathname);
    if (!entry) {
        response.writeHead(404);
        response.end('Not found');
        return;
    }
    const [filename, contentType] = entry;
    try {
        const stat = fs.statSync(filename);
        response.writeHead(200, {
            'Content-Type': contentType,
            'Content-Length': stat.size,
            'Cache-Control': 'no-store'
        });
        fs.createReadStream(filename).pipe(response);
    } catch (error) {
        response.writeHead(500);
        response.end(String(error));
    }
});

await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
});
const address = server.address();
if (!address || typeof address === 'string') throw new Error('Could not bind local renderer server');
const first = plan.views[0];
const query = new URLSearchParams({
    width: String(plan.width),
    height: String(plan.height),
    eye: first.eye.join(','),
    target: plan.target.join(','),
    up: plan.up.join(','),
    fov: String(plan.vertical_fov_deg),
    geometry: plan.geometry_renderer === 'shared-gaussian-moments' ? '1' : '0'
});
const url = `http://127.0.0.1:${address.port}/?${query}`;

fs.mkdirSync(outputDir, { recursive: true });
const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    args: [
        '--disable-gpu-sandbox',
        '--use-angle=metal',
        '--enable-unsafe-webgpu',
        '--hide-scrollbars'
    ]
});

try {
    const page = await browser.newPage();
    await page.setViewport({
        width: plan.width,
        height: plan.height,
        deviceScaleFactor: 1
    });
    page.on('console', message => {
        if (message.type() === 'error' || message.type() === 'warning') {
            console.error(`[browser:${message.type()}] ${message.text()}`);
        }
    });
    page.on('pageerror', error => console.error(`[browser:error] ${error.stack || error}`));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(
        () => window.renderReady === true || window.renderError,
        { timeout: 120000, polling: 100 }
    );
    const initial = await page.evaluate(() => ({ ready: window.renderReady, error: window.renderError }));
    if (!initial.ready) throw new Error(`3DGS load failed: ${initial.error}`);
    const canvas = await page.$('#application-canvas');
    if (!canvas) throw new Error('Renderer canvas was not found');

    for (let index = 0; index < plan.views.length; index += 1) {
        const view = plan.views[index];
        await page.evaluate(
            async (nextEye, nextTarget, nextUp, nextFov) => {
                await window.setCamera(nextEye, nextTarget, nextUp, nextFov);
            },
            view.eye,
            view.target ?? plan.target,
            view.up ?? plan.up,
            view.vertical_fov_deg ?? plan.vertical_fov_deg
        );
        const output = path.join(outputDir, view.image_name);
        await canvas.screenshot({ path: output, type: 'png' });
        if (plan.geometry_renderer === 'shared-gaussian-moments') {
            const encoded = await page.evaluate(() => window.captureGeometry());
            fs.writeFileSync(path.join(outputDir, view.image_name.replace(/\.png$/, '.moments.f32')),
                Buffer.from(encoded, 'base64'));
        }
        console.log(`[${String(index + 1).padStart(3, '0')}/${String(plan.views.length).padStart(3, '0')}] ${view.image_name}`);
    }
} finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
}
