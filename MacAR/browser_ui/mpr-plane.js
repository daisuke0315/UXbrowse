// Render an immutable world slice using this image's camera projection.
// Keep the existing 2D MPR canvas for fixed/photo picking and compositing.
export class MPRPlane {
    constructor(){this.canvas=document.createElement('canvas');this.gl=null;this.key=null;}
    initialize(){
        const gl=this.canvas.getContext('webgl',{alpha:true,antialias:false,premultipliedAlpha:false,preserveDrawingBuffer:true});
        if(!gl)throw Error('MPRのGPU投影を利用できません');
        const shader=(kind,source)=>{
            const s=gl.createShader(kind);gl.shaderSource(s,source);gl.compileShader(s);
            if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));
            return s;
        };
        const program=gl.createProgram();
        gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec4 position; attribute vec2 uv; varying vec2 texCoord; void main(){gl_Position=position; texCoord=uv;}'));
        gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float; varying vec2 texCoord; uniform sampler2D image; void main(){gl_FragColor=texture2D(image,texCoord);}'));
        gl.linkProgram(program);
        if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
        gl.useProgram(program);
        this.position=gl.getAttribLocation(program,'position');this.uv=gl.getAttribLocation(program,'uv');
        this.positions=gl.createBuffer();this.coords=gl.createBuffer();this.texture=gl.createTexture();
        gl.bindBuffer(gl.ARRAY_BUFFER,this.coords);
        gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([0,0,1,0,0,1,1,1]),gl.STATIC_DRAW);
        gl.enableVertexAttribArray(this.uv);gl.vertexAttribPointer(this.uv,2,gl.FLOAT,false,0,0);
        gl.bindTexture(gl.TEXTURE_2D,this.texture);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        this.gl=gl;
    }
    reset(){this.key=null;}
    draw(target,bitmap,quad,key,width,height){
        target.clearRect(0,0,width,height);
        if(!quad)return;
        if(!this.gl)this.initialize();
        const gl=this.gl;
        if(gl.isContextLost()){this.key=null;throw Error('MPRのGPUコンテキストが失われました');}
        if(bitmap){
            gl.bindTexture(gl.TEXTURE_2D,this.texture);
            gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,bitmap);
            // Corners refer to CT texel centers, matching OpenCV's convention.
            const x=.5/bitmap.width,y=.5/bitmap.height;
            gl.bindBuffer(gl.ARRAY_BUFFER,this.coords);
            gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([x,y,1-x,y,x,1-y,1-x,1-y]),gl.STATIC_DRAW);
            this.key=key;
        }
        if(this.key!==key)return;
        this.canvas.width=width;this.canvas.height=height;
        gl.viewport(0,0,width,height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindBuffer(gl.ARRAY_BUFFER,this.positions);
        gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([0,1,3,2].flatMap(i=>quad[i])),gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(this.position);gl.vertexAttribPointer(this.position,4,gl.FLOAT,false,0,0);
        gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
        target.drawImage(this.canvas,0,0);
    }
}
