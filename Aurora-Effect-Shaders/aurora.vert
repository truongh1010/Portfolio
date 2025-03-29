#version 330 compatibility

out vec3  vMCposition;
out vec3  vNormal; //
out float vLightIntensity;
out vec2  vST; 
out vec4  vColor;


vec3 LIGHTPOS = vec3(-2., 0., 10.);

void main() {
    vST = gl_MultiTexCoord0.st;

    vec3 tnorm = normalize(gl_NormalMatrix * gl_Normal);
    vNormal = tnorm;
    vec3 ECposition = vec3(gl_ModelViewMatrix * gl_Vertex);
    vLightIntensity = abs(dot(normalize(LIGHTPOS - ECposition), tnorm));


    vColor = vec4(1., 1., 1., 1.);
    vMCposition  = gl_Vertex.xyz;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
