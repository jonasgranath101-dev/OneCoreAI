/*

    Bindings for core files.

*/

// Header Guard (prevents double inclusion)
#ifndef HANDLE_H
#define HANDLE_H

// Function prototypes

// Learning logic function
int learn_logic();

// User interface functions
void learn();
void status();
void info();
void fetch_data();

// Block management functions
void block_clear();
void block_run();
void block_delete();
void block_status();
void block_config();

#endif
