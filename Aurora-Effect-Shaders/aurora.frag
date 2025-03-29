#version 330 compatibility

in vec3 vMCposition; // Position in model coordinates from vertex shader
in vec3 vNormal; // Normal from vertex shader
in float vLightIntensity; // Light intensity from vertex shader
in vec2 vST; // Texture coordinates from vertex shader
in vec4 vColor; // Color from vertex shader

uniform sampler2D uBackgroundTex; // Background texture uniform
uniform float Timer; // Time uniform for animation
uniform float uNoiseScale; // Noise scale for aurora effect
uniform float uTimeScale; // Time scale for aurora effect
uniform float uFrequency; // Controls the frequency of the wave pattern
uniform float uWaveSpeed; // Controls the speed of the wave movement
uniform float uIntensity;

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;


// Hash function used for generating pseudo-random values
float hash(vec2 p) {
    return fract(sin(dot(p.xy, vec2(12.9898,78.233))) * 43758.5453);
}

// Simple 2D noise function
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Four corners in 2D of a tile
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    // Smooth Interpolation
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a, b, u.x) +
           (c - a)* u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}


void main() {



    vec4 background = texture(uBackgroundTex, vST);
    float time = mod(Timer * uTimeScale, TWO_PI);

    float lowerLimit = 0.; // Start of the aurora effect
    float upperLimit = 1.0; // End of the aurora effect

    // Calculate the wave pattern
    float wave = sin(vMCposition.y * uFrequency + time * uWaveSpeed);

    // Apply noise based on the modulated position by the wave pattern
    float noiseValue = noise((vMCposition.xy + wave) * uNoiseScale);

    // Intensify the noise contribution for a more vivid effect
    float gradient = smoothstep(lowerLimit, upperLimit, vST.y) * (uIntensity + noiseValue);
    //float gradient = smoothstep(0.0, 1.0, vST.y) * uIntensity * noiseValue;

    //===========================================
    // Define base colors for the aurora with strong blue
    vec3 intenseBlue = vec3(0.0, gradient, gradient * 2.0);

    // Define other colors for the aurora
    vec3 green = vec3(0.0, 1.0, 0.0);
    vec3 purple = vec3(0.5, 0.0, 1.0);

    // Interpolate between colors based on vertical position and noise
    float positionFactor = vST.y; // Factor based on vertical position for color blending
    vec3 colorMix = intenseBlue;

    // Blend in other colors smoothly based on position and noise
    colorMix = mix(colorMix, green, smoothstep(0.3, 0.5, positionFactor) * noiseValue);
    colorMix = mix(colorMix, purple, smoothstep(0.6, 0.8, positionFactor) * (1.0 - noiseValue));
    // Apply the final color mix
    vec4 finalColor = vec4(colorMix, 0.8); // Slightly transparent
    //============================================

    // Increase the color saturation for a more vivid aurora
    //vec3 auroraColor = vec3(0.0, gradient, gradient * 2.0); // Make blue more intense


    // Apply the aurora color, ensuring it's fully visible
    //vec4 finalColor = vec4(auroraColor, 0.8);

    // Apply the aurora effect only within a specified range
    if (vST.y > lowerLimit && vST.y < upperLimit) {
        gl_FragColor = mix(background, finalColor, gradient);
    } else {
        gl_FragColor = background;
    }
}

