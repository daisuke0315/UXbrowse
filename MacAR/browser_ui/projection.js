// Column-major OpenGL projection, for an OpenCV K and right/up/back camera.
export function intrinsicProjection(K, width, height, near=.1, far=100000) {
    // CV integer coordinates denote pixel centers; canvas edges are at 0,w.
    return [2*K[0][0]/width,0,0,0, -2*K[0][1]/width,2*K[1][1]/height,0,0,
        1-2*(K[0][2]+.5)/width,2*(K[1][2]+.5)/height-1,-(far+near)/(far-near),-1,
        0,0,-2*far*near/(far-near),0];
}
