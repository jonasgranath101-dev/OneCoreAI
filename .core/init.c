#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Bindings.

#include "handle.h"

/*

    OneCoreAI - Multiple AI Core Blocks System

    Each core contains AI logic blocks with extractable variables.

*/

// Configuration variables
#define MAX_CORES 10
#define MAX_ITERATIONS 100
#define DATA_SIZE 1000

// AICore structure defined in handle.h

// Data structure for training samples
typedef struct {
    float x;
    float y;
} TrainingData;

// Global cores array
AICore cores[MAX_CORES];
int active_cores = 0;

// AI Block Functions - Core Logic Components

// Forward pass block: prediction = w * x + b
float ai_block_forward(float w, float b, float x) {
    return w * x + b;
}

// Loss calculation block: Mean Squared Error
float ai_block_loss(float prediction, float target) {
    float error = prediction - target;
    return error * error;
}

// Gradient calculation block
void ai_block_gradients(float prediction, float target, float x, float *dw, float *db) {
    float error = prediction - target;
    *dw = 2.0f * error * x;
    *db = 2.0f * error;
}

// Parameter update block
void ai_block_update(float *w, float *b, float dw, float db, float learning_rate) {
    *w -= learning_rate * dw;
    *b -= learning_rate * db;
}

// Training block - combines all AI blocks for one core
int ai_block_train(AICore *core, TrainingData *data, size_t data_size) {
    printf("Training Core %d (%s)...\n", core->id, core->name);

    // Reset loss history
    core->loss_count = 0;

    for (int epoch = 0; epoch < core->epochs; epoch++) {
        float total_loss = 0.0f;
        float avg_dw = 0.0f;
        float avg_db = 0.0f;

        // Forward pass and gradient accumulation
        for (size_t i = 0; i < data_size; i++) {
            float pred = ai_block_forward(core->weight, core->bias, data[i].x);
            total_loss += ai_block_loss(pred, data[i].y);

            float dw, db;
            ai_block_gradients(pred, data[i].y, data[i].x, &dw, &db);
            avg_dw += dw;
            avg_db += db;
        }

        // Average gradients and loss
        avg_dw /= data_size;
        avg_db /= data_size;
        total_loss /= data_size;

        // Update parameters
        ai_block_update(&core->weight, &core->bias, avg_dw, avg_db, core->learning_rate);

        // Store loss history
        if (epoch < 100) {
            core->loss_history[epoch] = total_loss;
            core->loss_count++;
        }

        // Print progress
        if ((epoch + 1) % 10 == 0) {
            printf("  Epoch %d: Loss = %.4f, w = %.4f, b = %.4f\n",
                   epoch + 1, total_loss, core->weight, core->bias);
        }
    }

    core->trained = 1;
    printf("Core %d training completed!\n", core->id);
    return 0;
}

// Prediction block
float ai_block_predict(AICore *core, float x) {
    if (!core->trained) {
        printf("Warning: Core %d not trained yet!\n", core->id);
        return 0.0f;
    }
    return ai_block_forward(core->weight, core->bias, x);
}

// Variable extraction blocks
void ai_block_extract_variables(AICore *core, float *w, float *b, float *lr, int *epochs) {
    *w = core->weight;
    *b = core->bias;
    *lr = core->learning_rate;
    *epochs = core->epochs;
}

void ai_block_load_variables(AICore *core, float w, float b, float lr, int epochs) {
    core->weight = w;
    core->bias = b;
    core->learning_rate = lr;
    core->epochs = epochs;
}

// Core Management Functions

// Create a new core
int core_create(const char *name, float learning_rate, int epochs) {
    if (active_cores >= MAX_CORES) {
        printf("Maximum cores reached!\n");
        return -1;
    }

    AICore *core = &cores[active_cores];
    core->id = active_cores + 1;
    strncpy(core->name, name, sizeof(core->name) - 1);
    core->weight = 0.0f;
    core->bias = 0.0f;
    core->learning_rate = learning_rate;
    core->epochs = epochs;
    core->trained = 0;
    core->loss_count = 0;

    printf("Created Core %d: %s\n", core->id, core->name);
    return active_cores++;
}

// Delete a core
void core_delete(int core_id) {
    if (core_id < 1 || core_id > active_cores) {
        printf("Invalid core ID!\n");
        return;
    }

    // Shift cores down
    for (int i = core_id - 1; i < active_cores - 1; i++) {
        cores[i] = cores[i + 1];
        cores[i].id = i + 1;
    }
    active_cores--;
    printf("Deleted Core %d\n", core_id);
}

// Get core by ID
AICore* core_get(int core_id) {
    if (core_id < 1 || core_id > active_cores) {
        return NULL;
    }
    return &cores[core_id - 1];
}

// Block functions for user interface (from handle.h)

// Clear block from variables.
void block_clear() {
    active_cores = 0;
    printf("All cores cleared.\n");
}

// Run a block (train a core).
void block_run() {
    if (active_cores == 0) {
        printf("No cores available. Create a core first.\n");
        return;
    }

    // Generate training data: y = 2*x + 1 + noise
    TrainingData *data = malloc(DATA_SIZE * sizeof(TrainingData));
    srand(time(NULL));

    for (size_t i = 0; i < DATA_SIZE; i++) {
        data[i].x = (float)i / 100.0f;  // Scale to 0-10 range
        data[i].y = 2.0f * data[i].x + 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    // Train all cores
    for (int i = 0; i < active_cores; i++) {
        ai_block_train(&cores[i], data, DATA_SIZE);
    }

    free(data);
}

// Delete a block.
void block_delete() {
    // For simplicity, delete the last core
    if (active_cores > 0) {
        core_delete(active_cores);
    }
}

// Display output of block activity.
void block_status() {
    printf("\n=== OneCoreAI Status ===\n");
    printf("Active Cores: %d\n\n", active_cores);

    for (int i = 0; i < active_cores; i++) {
        AICore *core = &cores[i];
        printf("Core %d (%s):\n", core->id, core->name);
        printf("  Trained: %s\n", core->trained ? "Yes" : "No");
        if (core->trained) {
            printf("  Weight: %.4f, Bias: %.4f\n", core->weight, core->bias);
            printf("  Learning Rate: %.4f, Epochs: %d\n", core->learning_rate, core->epochs);
            if (core->loss_count > 0) {
                printf("  Final Loss: %.4f\n", core->loss_history[core->loss_count - 1]);
            }
        }
        printf("\n");
    }
}

// Change block variables.
void block_config() {
    // For simplicity, reconfigure the first core
    if (active_cores > 0) {
        AICore *core = &cores[0];
        core->learning_rate = 0.02f;  // Example change
        core->epochs = 200;
        printf("Reconfigured Core %d\n", core->id);
    }
}

// Learn machine blocks.
void learn(float x, float y) {
    // This would train on a single sample - simplified
    if (active_cores > 0) {
        AICore *core = &cores[0];
        float pred = ai_block_forward(core->weight, core->bias, x);
        float dw, db;
        ai_block_gradients(pred, y, x, &dw, &db);
        ai_block_update(&core->weight, &core->bias, dw, db, core->learning_rate);
    }
}

// Fetch learned variables.
void fetch_data() {
    if (active_cores == 0) {
        printf("No cores available.\n");
        return;
    }

    // Extract variables from first core
    float w, b, lr;
    int epochs;
    ai_block_extract_variables(&cores[0], &w, &b, &lr, &epochs);
    printf("Core 1 Variables: w=%.4f, b=%.4f, lr=%.4f, epochs=%d\n", w, b, lr, epochs);
}

// Program diagnostic functions.
void status() {
    block_status();
}

void info() {
    printf("\n=== OneCoreAI Information ===\n");
    printf("Block-based AI system with multiple cores.\n");
    printf("Each core contains AI logic blocks with extractable variables.\n");
    printf("Commands: create cores, train, predict, extract variables.\n");
    printf("Maximum cores: %d\n", MAX_CORES);
}

int main(int argc, char *argv[]) {
    printf("Welcome to OneCoreAI - Multiple AI Core Blocks System\n\n");

    // Example usage: Create multiple cores
    int core1 = core_create("LinearRegression1", 0.01f, 100);
    int core2 = core_create("LinearRegression2", 0.02f, 50);
    int core3 = core_create("LinearRegression3", 0.005f, 200);

    if (core1 >= 0 && core2 >= 0 && core3 >= 0) {
        printf("\nCreated 3 AI cores. Running training...\n\n");

        // Run training on all cores
        block_run();

        printf("\nTesting predictions...\n");
        for (int i = 0; i < 3; i++) {
            AICore *core = &cores[i];
            float test_x = 5.0f;
            float pred = ai_block_predict(core, test_x);
            float true_y = 2.0f * test_x + 1.0f;
            printf("Core %d: x=%.1f, Predicted=%.4f, True=%.4f\n",
                   core->id, test_x, pred, true_y);
        }

        // Extract variables from cores
        printf("\nExtracting variables from cores...\n");
        for (int i = 0; i < 3; i++) {
            float w, b, lr;
            int epochs;
            ai_block_extract_variables(&cores[i], &w, &b, &lr, &epochs);
            printf("Core %d: w=%.4f, b=%.4f\n", cores[i].id, w, b);
        }
    }

    printf("\nOneCoreAI demonstration completed.\n");
    return 0;
}
