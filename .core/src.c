/*

    OneCoreAI - Additional AI Block Functions

    Extended logic & algorithms for data training and core operations.

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "handle.h"

// External reference to cores (defined in init.c)
extern AICore cores[];
extern int active_cores;

// Advanced AI Block Functions

// Batch normalization block (simplified)
void ai_block_batch_norm(float *data, size_t size, float *mean, float *variance) {
    *mean = 0.0f;
    *variance = 0.0f;

    // Calculate mean
    for (size_t i = 0; i < size; i++) {
        *mean += data[i];
    }
    *mean /= size;

    // Calculate variance
    for (size_t i = 0; i < size; i++) {
        float diff = data[i] - *mean;
        *variance += diff * diff;
    }
    *variance /= size;

    // Normalize
    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] - *mean) / sqrtf(*variance + 1e-8f);
    }
}

// Regularization block (L2)
float ai_block_l2_regularization(float w, float b, float lambda) {
    return lambda * (w * w + b * b);
}

// Learning rate decay block
float ai_block_lr_decay(float initial_lr, int epoch, float decay_rate) {
    return initial_lr * expf(-decay_rate * epoch);
}

// Cross-validation block
float ai_block_cross_validate(float (*predict_func)(float), float *x_test, float *y_test, size_t test_size) {
    float total_error = 0.0f;
    for (size_t i = 0; i < test_size; i++) {
        float pred = predict_func(x_test[i]);
        float error = pred - y_test[i];
        total_error += error * error;
    }
    return total_error / test_size;
}

// Save core variables to file
int ai_block_save_to_file(int core_id, const char *filename) {
    if (core_id < 1 || core_id > active_cores) {
        return -1;
    }

    FILE *file = fopen(filename, "w");
    if (!file) {
        return -1;
    }

    AICore *core = &cores[core_id - 1];
    fprintf(file, "Core Variables\n");
    fprintf(file, "ID: %d\n", core->id);
    fprintf(file, "Name: %s\n", core->name);
    fprintf(file, "Weight: %.6f\n", core->weight);
    fprintf(file, "Bias: %.6f\n", core->bias);
    fprintf(file, "Learning_Rate: %.6f\n", core->learning_rate);
    fprintf(file, "Epochs: %d\n", core->epochs);
    fprintf(file, "Trained: %d\n", core->trained);

    // Save loss history
    fprintf(file, "Loss_History_Count: %d\n", core->loss_count);
    for (int i = 0; i < core->loss_count; i++) {
        fprintf(file, "Loss_%d: %.6f\n", i, core->loss_history[i]);
    }

    fclose(file);
    return 0;
}

// Load core variables from file
int ai_block_load_from_file(int core_id, const char *filename) {
    if (core_id < 1 || core_id > active_cores) {
        return -1;
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        return -1;
    }

    AICore *core = &cores[core_id - 1];
    char line[256];

    while (fgets(line, sizeof(line), file)) {
        char key[64];
        float value;

        if (sscanf(line, "Weight: %f", &value) == 1) {
            core->weight = value;
        } else if (sscanf(line, "Bias: %f", &value) == 1) {
            core->bias = value;
        } else if (sscanf(line, "Learning_Rate: %f", &value) == 1) {
            core->learning_rate = value;
        } else if (sscanf(line, "Epochs: %d", &core->epochs) == 1) {
            // epochs is int
        } else if (sscanf(line, "Trained: %d", &core->trained) == 1) {
            // trained is int
        }
    }

    fclose(file);
    return 0;
}

// Ensemble prediction block (average predictions from multiple cores)
float ai_block_ensemble_predict(float x, int *core_ids, int num_cores) {
    if (num_cores == 0) return 0.0f;

    float total_pred = 0.0f;
    int valid_cores = 0;

    for (int i = 0; i < num_cores; i++) {
        int core_id = core_ids[i];
        if (core_id >= 1 && core_id <= active_cores) {
            AICore *core = &cores[core_id - 1];
            if (core->trained) {
                total_pred += core->weight * x + core->bias;
                valid_cores++;
            }
        }
    }

    return valid_cores > 0 ? total_pred / valid_cores : 0.0f;
}

// Calculate loss statistics across training history
void ai_block_loss_statistics(int core_id, float *min_loss, float *max_loss, float *avg_loss) {
    if (core_id < 1 || core_id > active_cores) {
        *min_loss = *max_loss = *avg_loss = 0.0f;
        return;
    }

    AICore *core = &cores[core_id - 1];
    if (core->loss_count == 0) {
        *min_loss = *max_loss = *avg_loss = 0.0f;
        return;
    }

    *min_loss = core->loss_history[0];
    *max_loss = core->loss_history[0];
    *avg_loss = 0.0f;

    for (int i = 0; i < core->loss_count; i++) {
        float loss = core->loss_history[i];
        if (loss < *min_loss) *min_loss = loss;
        if (loss > *max_loss) *max_loss = loss;
        *avg_loss += loss;
    }

    *avg_loss /= core->loss_count;
}

// Detect loss convergence
int ai_block_loss_converged(int core_id, float tolerance) {
    if (core_id < 1 || core_id > active_cores) {
        return 0;
    }

    AICore *core = &cores[core_id - 1];
    if (core->loss_count < 10) {  // Need minimum history
        return 0;
    }

    // Check if loss change is below tolerance for recent epochs
    float recent_change = 0.0f;
    int check_epochs = 5;
    if (check_epochs > core->loss_count) check_epochs = core->loss_count;

    for (int i = 1; i < check_epochs; i++) {
        int curr_idx = core->loss_count - i;
        int prev_idx = core->loss_count - i - 1;
        if (curr_idx >= 0 && prev_idx >= 0) {
            float change = (core->loss_history[prev_idx] - core->loss_history[curr_idx]) 
                          / (core->loss_history[prev_idx] + 1e-8f);
            if (change > recent_change) recent_change = change;
        }
    }

    return recent_change < tolerance;
}

// Compute loss gradient norm for stability analysis
float ai_block_loss_gradient_norm(float prediction, float target, float x, 
                                  LossType loss_type, float delta) {
    float dw, db;
    float w_dummy = 0.0f, b_dummy = 0.0f;
    
    // Get gradients
    ai_block_gradients_advanced(prediction, target, x, w_dummy, b_dummy, 
                               &dw, &db, loss_type, delta, 0.0f);
    
    // Return L2 norm of gradient
    return sqrtf(dw * dw + db * db);
}

// Legacy learn_logic function (for compatibility)
int learn_logic() {
    const size_t N = 1000;
    const int epochs = 100;
    const float learning_rate = 0.01f;

    float w = 0.0f, b = 0.0f;

    // Generate training data
    float *x_data = malloc(N * sizeof(float));
    float *y_data = malloc(N * sizeof(float));

    srand(time(NULL));
    for (size_t i = 0; i < N; i++) {
        x_data[i] = (float)i / 100.0f;
        y_data[i] = 2.0f * x_data[i] + 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    // Training loop
    printf("Legacy AI Training: Linear Regression\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f, dw = 0.0f, db = 0.0f;

        for (size_t i = 0; i < N; i++) {
            float pred = w * x_data[i] + b;
            total_loss += (pred - y_data[i]) * (pred - y_data[i]);

            float error = pred - y_data[i];
            dw += 2.0f * error * x_data[i];
            db += 2.0f * error;
        }

        dw /= N;
        db /= N;
        total_loss /= N;

        w -= learning_rate * dw;
        b -= learning_rate * db;

        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %d: Loss = %.4f, w = %.4f, b = %.4f\n", epoch + 1, total_loss, w, b);
        }
    }

    printf("Legacy training completed: w = %.4f, b = %.4f\n", w, b);

    free(x_data);
    free(y_data);
    return 0;
}
