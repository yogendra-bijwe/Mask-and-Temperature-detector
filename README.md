# Mask-and-Temperature-detector
This project is made to avoid customers/people who are not wearing mask. The system checks people for mask and temperature at the entrance before opening the door.   

Using Python and Machine Learning the program is able to distinguish between people wearing mask and people not wearing mask. 
In the "using-azure-iot" branch a file named "iot-mask-detect.py" has the implementation of Azure IoT Hub and Blob storage. 

Raspberry Pi 4 B is used as the CPU that runs this python program. It is connected with Pi camera for video capture, temperature sensor, servo motor that acts as a gate and LEDs to indicate mask status.

For Training the ML model I am using MobileNetV2 which is a very small and efficient CNN. You can learn about MobileNetV2 from this link- https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c

The Machine Learning model training code is well explained in this YouTube video - https://www.youtube.com/watch?v=Ax6P93r32KU&t=982s 

**Algorithm**
1. Person places their hand near the IR sensor
2. When hand is close enough Temp sensor will check their body temp
3. Simultaneously the camera detects mask on their face
4. The gate opens only if their body temp within limits and wearing mask
5. Buzzer goes off if body temp above normal detected
6. For all other cases gate doesn't open execpt [4] 

**Circuit Diagram**
![gate_temp_ir](https://user-images.githubusercontent.com/50228728/118763542-a56db380-b895-11eb-8dc5-95f16776b63f.jpg)

**Real-Life Representation**
Here is the real-life project image made by me.
![final-box-hanging](https://user-images.githubusercontent.com/50228728/118765998-79543180-b899-11eb-9b9e-c1cb3af2a93e.jpg)

**Live working Demo**
https://www.youtube.com/watch?v=qcFSD_tFuL4&t=10s


