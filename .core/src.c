/*

    AI Data Trainer: Linear Regression with Gradient Descent

    Logic & algorithms for data training.

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// Algorithm functions

// Forward pass: prediction = w * x + b
float forward(float w, float b, float x) {
    return w * x + b;
}

// Mean Squared Error loss
float mse_loss(float pred, float target) {
    return (pred - target) * (pred - target);
}

// Gradient descent update
void update_parameters(float *w, float *b, float dw, float db, float learning_rate) {
    *w -= learning_rate * dw;
    *b -= learning_rate * db;
}

int learn_logic(int w, float b) {
    const size_t N = 1000;  // Number of training samples
    const int epochs = 100;  // Number of training epochs
    const float learning_rate = 0.01f;

    // Initialize parameters
    float w = 0.0f;  // Weight
    float b = 0.0f;  // Bias

    // Generate training data: y = 2*x + 1 + noise
    float *x_data = malloc(N * sizeof(float));
    float *y_data = malloc(N * sizeof(float));

    srand(time(NULL));
    for (size_t i = 0; i < N; i++) {
        x_data[i] = (float)i / 100.0f;  // Scale to 0-10 range
        y_data[i] = 2.0f * x_data[i] + 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;  // Add noise
    }

    // Training loop
    printf("Starting AI Data Training: Linear Regression\n");
    printf("Target function: y = 2*x + 1 (with noise)\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        float dw = 0.0f;
        float db = 0.0f;

        // Forward pass and accumulate gradients
        for (size_t i = 0; i < N; i++) {
            float pred = forward(w, b, x_data[i]);
            total_loss += mse_loss(pred, y_data[i]);

            // Gradients
            float error = pred - y_data[i];
            dw += 2.0f * error * x_data[i];
            db += 2.0f * error;
        }

        // Average gradients
        dw /= N;
        db /= N;
        total_loss /= N;

        // Update parameters
        update_parameters(&w, &b, dw, db, learning_rate);

        // Print progress every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %d: Loss = %.4f, w = %.4f, b = %.4f\n", epoch + 1, total_loss, w, b);
        }
    }

    printf("\nTraining completed!\n");
    printf("Learned parameters: w = %.4f, b = %.4f\n", w, b);
    printf("Target was: w = 2.0, b = 1.0\n");

    // Test on new data
    printf("\nTesting on new samples:\n");
    for (int i = 0; i < 5; i++) {
        float test_x = (float)(i * 2) / 100.0f;
        float pred = forward(w, b, test_x);
        float true_y = 2.0f * test_x + 1.0f;
        printf("x = %.2f: Predicted = %.4f, True = %.4f\n", test_x, pred, true_y);
    }

    // Clean up
    free(x_data);
    free(y_data);

    return 0;
}