#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
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
#define MAX_CORES 30
#define MAX_ITERATIONS 100
#define DATA_SIZE 1000
#define DISK_SIZE 100

// AICore structure defined in handle.h

// Data structure for training samples
typedef struct {
    unsigned char data_sheet;  // Hexadecimal data sheet for logic control
    float x;
    float y;
} TrainingData;

// Global cores array
AICore cores[MAX_CORES];
int active_cores = 0;

// Global hex data storage for recent training
#define MAX_HEX_DATA 1000
unsigned char recent_hex_data[MAX_HEX_DATA];
int recent_hex_count = 0;

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

// Function to visualize a core's variables as containers
void visualize_core(AICore *core, float current_loss) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║                     Core %d: %s                      ║\n", core->id, core->name);
    printf("╠══════════════════════════════════════════════════════════╣\n");

    // Weight visualization
    int weight_bars = (int)(core->weight * 5); // Scale for visualization
    if (weight_bars < 0) weight_bars = 0;
    if (weight_bars > 20) weight_bars = 20;
    printf("║ Weight:  [");
    for (int i = 0; i < 20; i++) {
        printf(i < weight_bars ? "█" : "░");
    }
    printf("] %.4f ║\n", core->weight);

    // Bias visualization
    int bias_bars = (int)(core->bias * 20); // Scale bias (0-1+)
    if (bias_bars < 0) bias_bars = 0;
    if (bias_bars > 20) bias_bars = 20;
    printf("║ Bias:    [");
    for (int i = 0; i < 20; i++) {
        printf(i < bias_bars ? "█" : "░");
    }
    printf("] %.4f ║\n", core->bias);

    // Learning rate visualization
    int lr_bars = (int)(core->learning_rate * 2000); // Scale lr (0.005-0.02)
    if (lr_bars < 0) lr_bars = 0;
    if (lr_bars > 20) lr_bars = 20;
    printf("║ LR:      [");
    for (int i = 0; i < 20; i++) {
        printf(i < lr_bars ? "█" : "░");
    }
    printf("] %.4f ║\n", core->learning_rate);

    // Loss visualization (inverse, lower loss = more filled)
    float loss_scale = current_loss > 1.0f ? 1.0f : current_loss;
    int loss_bars = (int)((1.0f - loss_scale) * 20); // Higher bars = lower loss
    if (loss_bars < 0) loss_bars = 0;
    if (loss_bars > 20) loss_bars = 20;
    printf("║ Loss:    [");
    for (int i = 0; i < 20; i++) {
        printf(i < loss_bars ? "█" : "░");
    }
    printf("] %.4f ║\n", current_loss);

    // Epoch progress
    int epoch_progress = (int)((float)core->epochs / 200 * 20); // Assuming max 200 epochs
    if (epoch_progress > 20) epoch_progress = 20;
    printf("║ Epochs:  [");
    for (int i = 0; i < 20; i++) {
        printf(i < epoch_progress ? "█" : "░");
    }
    printf("] %d/%d ║\n", core->epochs, 200);

    printf("╚══════════════════════════════════════════════════════════╝\n");
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

            // Apply hexadecimal data sheet logic to gradients
            unsigned char hex = data[i].data_sheet;
            if (hex & 0x01) dw *= 2.0f;  // Bit 0: Amplify weight gradient
            if (hex & 0x02) db *= 2.0f;  // Bit 1: Amplify bias gradient
            if (hex & 0x04) dw = -dw;    // Bit 2: Invert weight gradient
            if (hex & 0x08) db = -db;    // Bit 3: Invert bias gradient
            if (hex & 0x10) {            // Bit 4: Scale gradients up
                dw *= 1.5f;
                db *= 1.5f;
            }
            if (hex & 0x20) {            // Bit 5: Scale gradients down
                dw *= 0.5f;
                db *= 0.5f;
            }
            if (hex & 0x40) {            // Bit 6: Swap gradients
                float temp = dw;
                dw = db;
                db = temp;
            }
            if (hex & 0x80) {            // Bit 7: Zero gradients
                dw = 0.0f;
                db = 0.0f;
            }

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

        // Visualize the core every 5 epochs
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            printf("\033[2J\033[H"); // Clear screen
            visualize_core(core, total_loss);
            printf("Epoch: %d/%d\n", epoch + 1, core->epochs);
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


// Block size on disk.

void block_size(int id_core) {

    AICore *core = core_get(id_core);

    // This command finds the size of 'target_function' in the compiled binary
    // Note: The binary must be compiled with symbols (not stripped)
    printf("Fetching function size from symbol table...\n");
    
    // Replace 'a.out' with your executable name
    system("nm --print-size --size-sort onecoreai | grep core");
}

// Block disk and hardware location.

void block_location(int id_arg) {

    AICore *core = core_get(id_arg);

    // This is the 'address' stored inside the pointer (Location of Data)
    printf("Core storage address: %p\n", (void*)core);
    
    // This is the location of the pointer itself (Location on Stack)
    printf("Core own address:  %p\n", (void*)&core);
}

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

    // Reset hex data storage
    recent_hex_count = 0;

    for (size_t i = 0; i < DATA_SIZE; i++) {
        data[i].x = (float)i / 100.0f;  // Scale to 0-10 range
        data[i].y = 2.0f * data[i].x + 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        data[i].data_sheet = (unsigned char)(rand() % 256);  // Generate hexadecimal data sheet

        // Store hex data for listing
        if (recent_hex_count < MAX_HEX_DATA) {
            recent_hex_data[recent_hex_count++] = data[i].data_sheet;
        }
    }

    // Train all cores
    for (int i = 0; i < active_cores; i++) {
        ai_block_train(&cores[i], data, DATA_SIZE);
    }

    free(data);
}

// Train specific cores
void train_cores(int num_cores, int *core_ids) {
    if (num_cores == 0) {
        printf("No cores to train.\n");
        return;
    }

    // Generate training data: y = 2*x + 1 + noise
    TrainingData *data = malloc(DATA_SIZE * sizeof(TrainingData));
    srand(time(NULL));

    // Reset hex data storage
    recent_hex_count = 0;

    for (size_t i = 0; i < DATA_SIZE; i++) {
        data[i].x = (float)i / 100.0f;  // Scale to 0-10 range
        data[i].y = 2.0f * data[i].x + 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        data[i].data_sheet = (unsigned char)(rand() % 256);  // Generate hexadecimal data sheet

        // Store hex data for listing
        if (recent_hex_count < MAX_HEX_DATA) {
            recent_hex_data[recent_hex_count++] = data[i].data_sheet;
        }
    }

    // Train specified cores
    for (int i = 0; i < num_cores; i++) {
        int core_id = core_ids[i];
        AICore *core = core_get(core_id);
        if (core) {
            ai_block_train(core, data, DATA_SIZE);
        } else {
            printf("Invalid core ID: %d\n", core_id);
        }
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

    // Block disk size configuration.



    // For simplicity, reconfigure the first core
    if (active_cores > 0) {
        AICore *core = &cores[0];
        core->learning_rate = 0.02f;  // Example change
        core->epochs = 200;
        printf("Reconfigured Core %d\n", core->id);
    }
}

// Learn machine blocks for specific core.
void learn(int core_id, float x, float y) {
    AICore *core = core_get(core_id);
    if (core) {
        float pred = ai_block_forward(core->weight, core->bias, x);
        float dw, db;
        ai_block_gradients(pred, y, x, &dw, &db);
        ai_block_update(&core->weight, &core->bias, dw, db, core->learning_rate);
        printf("Trained Core %d on sample (%.2f, %.2f)\n", core_id, x, y);
    } else {
        printf("Invalid core ID: %d\n", core_id);
    }
}

// Fetch learned variables from specific core.
void fetch_data(int core_id) {
    AICore *core = core_get(core_id);
    if (core) {
        float w, b, lr;
        int epochs;
        ai_block_extract_variables(core, &w, &b, &lr, &epochs);
        printf("Core %d Variables: w=%.4f, b=%.4f, lr=%.4f, epochs=%d\n", core_id, w, b, lr, epochs);
    } else {
        printf("Invalid core ID: %d\n", core_id);
    }
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

// Display hexadecimal data list from recent training
void hex_list() {
    printf("\n=== Recent Training Hex Data ===\n");
    printf("Hex values used in the last training session:\n\n");

    if (recent_hex_count == 0) {
        printf("No recent training data available.\n");
        printf("Run 'run' or 'train <core_id>' to generate hex data.\n");
        return;
    }

    printf("Total hex values: %d\n\n", recent_hex_count);

    // Display in rows of 16 for readability
    int items_per_row = 16;
    for (int i = 0; i < recent_hex_count; i++) {
        printf("%02X ", recent_hex_data[i]);
        if ((i + 1) % items_per_row == 0) {
            printf("\n");
        }
    }
    if (recent_hex_count % items_per_row != 0) {
        printf("\n");
    }

    printf("\nHex data affects gradient computation during training:\n");
    printf("Bit 0: Amplify weight gradient\n");
    printf("Bit 1: Amplify bias gradient\n");
    printf("Bit 2: Invert weight gradient\n");
    printf("Bit 3: Invert bias gradient\n");
    printf("Bit 4: Scale gradients up\n");
    printf("Bit 5: Scale gradients down\n");
    printf("Bit 6: Swap weight and bias gradients\n");
    printf("Bit 7: Zero gradients\n");
}

int main(int argc, char *argv[]) {
    printf("Welcome to OneCoreAI - Multiple AI Core Blocks System\n");
    printf("Type 'help' for available commands.\n\n");

    char command[256];
    char arg1[64], arg2[64], arg3[64], arg4[64];

    while (1) {
        printf("OneCoreAI> ");
        fflush(stdout);

        if (fgets(command, sizeof(command), stdin) == NULL) {
            break;
        }

        // Remove newline character
        command[strcspn(command, "\n")] = 0;

        // Parse command and arguments
        int args_count = sscanf(command, "%s %s %s %s", arg1, arg2, arg3, arg4);

        if (strcmp(arg1, "exit") == 0 || strcmp(arg1, "quit") == 0) {
            break;
        } else if (strcmp(arg1, "help") == 0) {
            printf("\nAvailable Commands:\n");
            printf("  create <name> <lr> <epochs>  - Create a new AI core\n");
            printf("  run                          - Train all cores (shows visualization)\n");
            printf("  status                       - Show status of all cores\n");
            printf("  predict <core_id> <x>        - Make prediction with specific core\n");
            printf("  delete <core_id>             - Delete a specific core\n");
            printf("  size <core_id>               - Disk block size.\n");
            printf("  location <core_id>           - Block disk location\n");
            printf("  clear                        - Clear all cores\n");
            printf("  config <core_id> <lr> <epochs> - Configure a core\n");
            printf("  train <core_id> [core_id2] ... - Train specific cores\n");
            printf("  learn <core_id> <x> <y>      - Train specific core on single sample\n");
            printf("  fetch <core_id>              - Extract variables from specific core\n");
            printf("  hexlist                      - Display hex data from recent training\n");
            printf("  info                         - Show system information\n");
            printf("  help                         - Show this help message\n");
            printf("  exit                         - Exit the program\n\n");
        } else if (strcmp(arg1, "create") == 0 && args_count >= 4) {
            float lr = atof(arg3);
            int epochs = atoi(arg4);
            core_create(arg2, lr, epochs);
        } else if (strcmp(arg1, "run") == 0) {
            block_run();
        } else if (strcmp(arg1, "status") == 0) {
            block_status();
        } else if (strcmp(arg1, "predict") == 0 && args_count >= 4) {
            int core_id = atoi(arg2);
            float x = atof(arg3);
            float y = atof(arg4);
            AICore *core = core_get(core_id);
            if (core) {
                float pred = ai_block_predict(core, x);
                printf("Core %d prediction for x=%.2f: %.4f\n", core_id, x, pred);
            } else {
                printf("Invalid core ID: %d\n", core_id);
            }
        } else if (strcmp(arg1, "delete") == 0 && args_count >= 2) {
            int core_id = atoi(arg2);
            core_delete(core_id);
        } else if (strcmp(arg1, "clear") == 0) {
            block_clear();
        } else if (strcmp(arg1, "location") == 0) {
            int core_id = atoi(arg2);
            block_location(core_id);
        } else if (strcmp(arg1, "size") == 0) {
            int core_id = atoi(arg2);
            block_size(core_id);
        } else if (strcmp(arg1, "config") == 0 && args_count >= 4) {
            int core_id = atoi(arg2);
            AICore *core = core_get(core_id);
            if (core) {
                core->learning_rate = atof(arg3);
                core->epochs = atoi(arg4);
                printf("Reconfigured Core %d: lr=%.4f, epochs=%d\n", core_id, core->learning_rate, core->epochs);
            } else {
                printf("Invalid core ID: %d\n", core_id);
            }
        } else if (strcmp(arg1, "train") == 0 && args_count >= 2) {
            int core_ids[30];
            int count = 0;
            if (args_count >= 2) core_ids[count++] = atoi(arg2);
            if (args_count >= 3) core_ids[count++] = atoi(arg3);
            if (args_count >= 4) core_ids[count++] = atoi(arg4);
            train_cores(count, core_ids);
        
        } else if (strcmp(arg1, "learn") == 0 && args_count >= 4) {
            int core_id = atoi(arg2);
            float x = atof(arg3);
            float y = atof(arg4);
            learn(core_id, x, y);
        } else if (strcmp(arg1, "fetch") == 0 && args_count >= 2) {
            fetch_data(atoi(arg2));
        } else if (strcmp(arg1, "hexlist") == 0) {
            hex_list();
        } else if (strcmp(arg1, "info") == 0) {
            info();
        } else if (strlen(arg1) > 0) {
            printf("Unknown command: %s\n", arg1);
            printf("Type 'help' for available commands.\n");
        }
        printf("\n");
    }

    printf("Goodbye!\n");
    return 0;
}
