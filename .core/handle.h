/*

    Bindings for core files.

*/

// Header Guard (prevents double inclusion)
#ifndef HANDLE_H
#define HANDLE_H

// AI Core structure - represents a single AI processing unit
typedef struct {
    int id;
    char name[32];
    float weight;        // Learned parameter w
    float bias;          // Learned parameter b
    float learning_rate;
    int epochs;
    int trained;         // Flag indicating if core has been trained
    float loss_history[100]; // Store loss over time
    int loss_count;
} AICore;

// Function prototypes

// Learning logic function
int learn_logic();

// User interface functions
void learn(int core_id, float x, float y);
void status();
void info();
void fetch_data(int core_id);

// Block management functions
void block_clear();
void block_run();
void block_delete();
void block_status();
void block_config();
void block_location(int core_id);

#endif
