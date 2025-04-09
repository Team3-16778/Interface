void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "LED_ON") {
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("ACK: LED turned on");
    } else {
      Serial.println("ERR: Unknown command");
    }
  }
}
