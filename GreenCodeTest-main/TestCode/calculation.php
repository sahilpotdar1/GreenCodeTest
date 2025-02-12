<?php
echo "Welcome to the Inefficient PHP Application!<br>";

// Unnecessary delay
sleep(2);

echo "Enter the first number: ";
$num1 = 10; // Simulating user input
echo $num1 . "<br>";

// Unnecessary loop
for ($i = 0; $i < 1000000; $i++) {
    $num1 += 0.000001;
}

echo "Enter the second number: ";
$num2 = 20; // Simulating user input
echo $num2 . "<br>";

// Unnecessary loop
for ($i = 0; $i < 1000000; $i++) {
    $num2 += 0.000001;
}

echo "Choose an operation:<br>";
echo "1. Addition<br>";
echo "2. Subtraction<br>";
echo "3. Multiplication<br>";
echo "4. Division<br>";

$choice = 1; // Simulating user input
echo "You chose: " . $choice . "<br>";

$result = 0;

// Inefficient data structure
$operations = array(
    1 => 'add',
    2 => 'subtract',
    3 => 'multiply',
    4 => 'divide'
);

if (array_key_exists($choice, $operations)) {
    $result = $operations$choice;
} else {
    echo "Invalid choice.<br>";
}

// Unnecessary delay
sleep(2);

echo "The result is: " . $result . "<br>";

function add($a, $b) {
    // Redundant computation
    for ($i = 0; $i < 1000000; $i++) {}
    return $a + $b;
}

function subtract($a, $b) {
    // Redundant computation
    for ($i = 0; $i < 1000000; $i++) {}
    return $a - $b;
}

function multiply($a, $b) {
    // Redundant computation
    for ($i = 0; $i < 1000000; $i++) {}
    return $a * $b;
}

function divide($a, $b) {
    // Redundant computation
    for ($i = 0; $i < 1000000; $i++) {}
    if ($b != 0) {
        return $a / $b;
    } else {
        echo "Cannot divide by zero.<br>";
        return 0;
    }
}
?>
