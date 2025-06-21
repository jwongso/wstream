// -------------------------------------------------------------------------------------------------
//
// Copyright (C) all of the contributors. All rights reserved.
//
// This software, including documentation, is protected by copyright controlled by
// contributors. All rights are reserved. Copying, including reproducing, storing,
// adapting or translating, any or all of this material requires the prior written
// consent of all contributors.
//
// -------------------------------------------------------------------------------------------------

#include "wstream_app.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

// Global shutdown flag
extern std::atomic<bool> g_shutdown_requested;

int main(int argc, char* argv[]) {
    try {
        wstream_app app;

        if (!app.initialize(argc, argv)) {
            std::cerr << "Failed to initialize application.\n";
            return 1;
        }

        app.run();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    // If we get here, shutdown was requested
    // Give it 3 seconds to clean up, then force exit
    for (int i = 0; i < 30; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!g_shutdown_requested) {
            return 0;  // Clean exit
        }
    }

    std::cout << "Shutdown timeout - forcing exit!" << std::endl;
    std::exit(0);
}
