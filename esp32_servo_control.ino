/*
  ESP32 Servo Control for Seed Sorting
  
  This code receives commands via Serial to control a servo motor:
  - "LEFT" - Move servo 90 degrees left (bad seeds)
  - "RIGHT" - Move servo 90 degrees right (good seeds)
  - "CENTER" - Move servo to center position
  
  Hardware connections:
  - Servo signal pin: GPIO 18
  - Servo VCC: 5V or 3.3V (depending on servo)
  - Servo GND: GND
*/

#include <ESP32Servo.h>

// Servo configuration
Servo sorterServo;
const int SERVO_PIN = 18;  // GPIO pin for servo control

// Servo positions (in degrees)
const int CENTER_POS = 90;   // Center position
const int LEFT_POS = 0;      // 90 degrees left (bad seeds)
const int RIGHT_POS = 180;   // 90 degrees right (good seeds)

// Auto-return timing
const int RETURN_DELAY = 1500;  // Time in ms before returning to center
unsigned long lastMoveTime = 0;
bool autoReturnEnabled = false;

// Current servo position
int currentPosition = CENTER_POS;

// Serial communication
String command = "";
bool commandComplete = false;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("ESP32 Servo Control for Seed Sorting - Starting...");
  
  // Attach servo to pin
  sorterServo.attach(SERVO_PIN);
  
  // Move servo to center position
  sorterServo.write(CENTER_POS);
  currentPosition = CENTER_POS;
  
  Serial.println("Servo initialized at center position");
  Serial.println("Ready to receive commands: LEFT, RIGHT, CENTER");
  
  // Reserve string space for commands
  command.reserve(20);
}

void loop() {
  // Check for incoming serial data
  if (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      commandComplete = true;
    } else {
      command += inChar;
    }
  }
  
  // Process complete command
  if (commandComplete) {
    processCommand(command);
    command = "";
    commandComplete = false;
  }
  
  // Auto-return to center after delay
  if (autoReturnEnabled && millis() - lastMoveTime >= RETURN_DELAY) {
    if (currentPosition != CENTER_POS) {
      moveServo(CENTER_POS, "CENTER (Auto-return)");
    }
    autoReturnEnabled = false;
  }
  
  delay(10);  // Small delay for stability
}

void processCommand(String cmd) {
  cmd.trim();  // Remove whitespace
  cmd.toUpperCase();  // Convert to uppercase
  
  Serial.print("Received command: ");
  Serial.println(cmd);
    if (cmd == "LEFT") {
    moveServo(LEFT_POS, "LEFT (Bad Seed)");
    autoReturnEnabled = true;
    lastMoveTime = millis();
  } 
  else if (cmd == "RIGHT") {
    moveServo(RIGHT_POS, "RIGHT (Good Seed)");
    autoReturnEnabled = true;
    lastMoveTime = millis();
  } 
  else if (cmd == "CENTER") {
    moveServo(CENTER_POS, "CENTER");
    autoReturnEnabled = false;  // Disable auto-return when manually centered
  }
  else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
    Serial.println("Valid commands: LEFT, RIGHT, CENTER");
  }
}

void moveServo(int targetPosition, String description) {
  if (targetPosition != currentPosition) {
    Serial.print("Moving servo to ");
    Serial.print(description);
    Serial.print(" (");
    Serial.print(targetPosition);
    Serial.println(" degrees)");
    
    // Smooth movement to target position
    int step = (targetPosition > currentPosition) ? 1 : -1;
    
    for (int pos = currentPosition; pos != targetPosition; pos += step) {
      sorterServo.write(pos);
      delay(15);  // Adjust speed of movement (lower = faster)
    }
    
    // Ensure we reach exact target position
    sorterServo.write(targetPosition);
    currentPosition = targetPosition;
    
    Serial.print("Servo positioned at ");
    Serial.print(targetPosition);
    Serial.println(" degrees");
  } else {
    Serial.print("Servo already at ");
    Serial.print(description);
    Serial.println(" position");
  }
}

void testSequence() {
  // Test function to verify servo movement
  Serial.println("Starting servo test sequence...");
  
  moveServo(CENTER_POS, "CENTER");
  delay(1000);
  
  moveServo(LEFT_POS, "LEFT");
  delay(1000);
  
  moveServo(CENTER_POS, "CENTER");
  delay(1000);
  
  moveServo(RIGHT_POS, "RIGHT");
  delay(1000);
  
  moveServo(CENTER_POS, "CENTER");
  delay(1000);
  
  Serial.println("Test sequence complete");
}

// Function to call from serial monitor for testing
// Send "TEST" to run test sequence
void serialEvent() {
  if (Serial.available()) {
    String testCmd = Serial.readString();
    testCmd.trim();
    testCmd.toUpperCase();
    
    if (testCmd == "TEST") {
      testSequence();
    }
  }
}
