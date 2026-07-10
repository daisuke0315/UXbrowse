/**
 * @typedef {[number, number, number]} Vector3
 * @typedef {{rows:number, columns:number, pixelSpacing:[number, number], imagePosition:Vector3, imageOrientation:[number, number, number, number, number, number], slicePosition:number, instanceNumber:number|null, seriesInstanceUid:string, studyInstanceUid:string, rescaleSlope:number, rescaleIntercept:number, windowCenter:number|null, windowWidth:number|null, bitsAllocated:number, bitsStored:number, pixelRepresentation:number, photometricInterpretation:string, transferSyntaxUid:string, pixelData:Int16Array|Uint16Array, fileName:string}} DicomSlice
 * @typedef {{dimensions:[number, number, number], origin:Vector3, spacing:Vector3, direction:number[], inverseDirection:number[], minimumValue:number, maximumValue:number, minimumHU:number, maximumHU:number, sourceFormat:"DICOM"|"MHA"|"DEMO", scalarType:string, seriesInstanceUid:string, studyInstanceUid:string, windowCenter:number|null, windowWidth:number|null}} VolumeMetadata
 * @typedef {{data:Int8Array|Uint8Array|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array, metadata:VolumeMetadata}} Volume
 * @typedef {{center:Vector3, u:Vector3, v:Vector3, n:Vector3, warnings:string[]}} MprPlane
 * @typedef {{width:number, height:number, spacingU:number, spacingV:number, eye:Vector3, target:Vector3, up:Vector3, normalOffset:number, interpolation:"linear"|"nearest", outsideValue:number, windowLevel:number, windowWidth:number}} MprRequest
 * @typedef {{pixels:Int16Array|Float32Array, rgba:Uint8ClampedArray, plane:MprPlane, width:number, height:number, elapsedMs:number, outsideCount:number, invalidCount:number}} MprResult
 */

export {};
