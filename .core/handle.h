/*

    Bindings for core files.

*/

// Header Guard (prevents double inclusion)
#ifndef HANDLE_H
#define HANDLE_H

// Loss function types
typedef enum {
    LOSS_MSE = 0,    // Mean Squared Error
    LOSS_MAE = 1,    // Mean Absolute Error
    LOSS_HUBER = 2   // Huber Loss (robust to outliers)
} LossType;

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
    LossType loss_type;  // Type of loss function to use
    float regularization_lambda;  // L2 regularization coefficient
    float huber_delta;   // Delta parameter for Huber loss
} AICore;

// Function prototypes

// Learning logic function
int learn_logic();

// AI Block Functions - Loss and Gradient Calculations
float ai_block_loss(float prediction, float target);
float ai_block_loss_mae(float prediction, float target);
float ai_block_loss_huber(float prediction, float target, float delta);
float ai_block_loss_with_regularization(float prediction, float target, float weight, 
                                       float bias, LossType loss_type, float delta, float lambda);

void ai_block_gradients(float prediction, float target, float x, float *dw, float *db);
void ai_block_gradients_advanced(float prediction, float target, float x, 
                                float weight, float bias, float *dw, float *db,
                                LossType loss_type, float delta, float lambda);

// Advanced Loss Analysis Functions
void ai_block_loss_statistics(int core_id, float *min_loss, float *max_loss, float *avg_loss);
int ai_block_loss_converged(int core_id, float tolerance);
float ai_block_loss_gradient_norm(float prediction, float target, float x, 
                                  LossType loss_type, float delta);

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
