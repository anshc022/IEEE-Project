/*
  ESP32 Seed Sorter - Simple Version
  
  Hardware Setup:
  - Circular box with cardboard divider
  - Good side: Seeds fall naturally (no servo movement)  
  - Bad side: Servo pushes/throws seeds to bad side
  - Servo pin: GPIO 18
  
  Commands:
  - "GOOD" - Do nothing (let seed fall to good side)
  - "BAD"  - Activate servo to throw seed to bad side
  - "TEST" - Test servo movement
*/

#include <ESP32Servo.h>

Servo seedSorter;
int servoPin = 18;

// Servo positions for your circular box - ADJUST THESE FOR YOUR SETUP
int REST_POSITION = 90;    // Default position (center) - good seeds fall naturally
int THROW_POSITION = 30;   // Position to throw bad seeds to bad side (adjust as needed)
int SWEEP_POSITION = 150;  // Optional sweep position (adjust as needed)

bool servoActive = false;
unsigned long servoStartTime = 0;
int throwDuration = 300;   // How long to hold throw position (milliseconds)

void setup() {
  Serial.begin(115200);
  Serial.println("=== ESP32 Seed Sorter v2.0 ===");
  Serial.println("Circular Box with Cardboard Divider");
  Serial.println("Commands: GOOD, BAD, TEST");
  
  // Initialize servo
  seedSorter.attach(servoPin, 500, 2400);
  seedSorter.write(REST_POSITION);
  
  Serial.println("✅ Servo ready at rest position");
  Serial.println("📦 Good seeds fall naturally, bad seeds get thrown!");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    processCommand(command);
  }
  
  // Handle servo timing for throw action
  if (servoActive && (millis() - servoStartTime > throwDuration)) {
    returnToRest();
  }
  
  delay(10);
}

void processCommand(String cmd) {
  Serial.print("Command received: ");
  Serial.println(cmd);
  
  if (cmd == "GOOD") {
    handleGoodSeed();
  }
  else if (cmd == "BAD") {
    handleBadSeed();
  }
  else if (cmd == "TEST") {
    testServo();
  }
  else if (cmd == "REST") {
    returnToRest();
  }
  else {
    Serial.println("❌ Unknown command. Use: GOOD, BAD, TEST, REST");
  }
}

void handleGoodSeed() {
  Serial.println("🌱 GOOD SEED - No action (falls to good side naturally)");
  // Do nothing - servo stays at rest position
  // Good seeds fall straight down to good side
}

void handleBadSeed() {
  Serial.println("🚫 BAD SEED - Throwing to bad side!");
  
  // Quick throw motion to push seed to bad side
  seedSorter.write(THROW_POSITION);
  servoActive = true;
  servoStartTime = millis();
  
  Serial.print("⚡ Servo active for ");
  Serial.print(throwDuration);
  Serial.println("ms");
}

void returnToRest() {
  Serial.println("🔄 Returning to rest position");
  seedSorter.write(REST_POSITION);
  servoActive = false;
}

void testServo() {
  Serial.println("🧪 Testing servo movement...");
  
  // Test sequence
  Serial.println("1. Rest position (90°)");
  seedSorter.write(REST_POSITION);
  delay(1000);
  
  Serial.println("2. Throw position (45°)");
  seedSorter.write(THROW_POSITION);
  delay(1000);
  
  Serial.println("3. Sweep position (135°)");
  seedSorter.write(SWEEP_POSITION);
  delay(1000);
  
  Serial.println("4. Back to rest");
  seedSorter.write(REST_POSITION);
  delay(500);
  
  Serial.println("✅ Test complete!");
}