# Cortex Neural Network Framework - Makefile
# Author: Barsbold
# Description: Build system for CPU-based neural network framework

# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -Wpedantic -std=c11 -O3 -march=native
DEBUG_CFLAGS = -Wall -Wextra -Wpedantic -std=c11 -O0 -g -fsanitize=address -fsanitize=undefined
INCLUDES = -Iinclude
LDFLAGS = -lm

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests
EXAMPLES_DIR = examples

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Test files
TEST_SRCS = $(wildcard $(TEST_DIR)/test_*.c)
TEST_BINS = $(TEST_SRCS:$(TEST_DIR)/test_%.c=$(BIN_DIR)/test_%)

# Example files
EXAMPLE_SRCS = $(wildcard $(EXAMPLES_DIR)/*.c)
EXAMPLE_BINS = $(EXAMPLE_SRCS:$(EXAMPLES_DIR)/%.c=$(BIN_DIR)/%)

# Library output
LIB_NAME = libcortex.a

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
RESET = \033[0m

# Default target
.PHONY: all
all: $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR)/$(LIB_NAME)
	@echo "$(GREEN)✓ Build complete!$(RESET)"
	@echo "$(YELLOW)Library: $(BUILD_DIR)/$(LIB_NAME)$(RESET)"

# Build library
$(BUILD_DIR)/$(LIB_NAME): $(OBJS)
	@echo "$(YELLOW)Creating static library...$(RESET)"
	ar rcs $@ $^
	@echo "$(GREEN)✓ Library created: $@$(RESET)"

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "$(YELLOW)Compiling $<...$(RESET)"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Build all tests
.PHONY: tests
tests: $(BUILD_DIR)/$(LIB_NAME) $(TEST_BINS)
	@echo "$(GREEN)✓ All tests built!$(RESET)"

# Build individual test executables
$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.c $(BUILD_DIR)/$(LIB_NAME) | $(BIN_DIR)
	@echo "$(YELLOW)Building test: $@...$(RESET)"
	$(CC) $(CFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcortex $(LDFLAGS) -o $@

# Run all tests
.PHONY: test
test: tests
	@echo "\n$(YELLOW)==============================$(RESET)"
	@echo "$(YELLOW)Running all tests...$(RESET)"
	@echo "$(YELLOW)==============================$(RESET)\n"
	@for test in $(TEST_BINS); do \
		echo "$(YELLOW)Running $$test...$(RESET)"; \
		./$$test || exit 1; \
		echo ""; \
	done
	@echo "$(GREEN)✓ All tests passed!$(RESET)\n"

# Build all examples
.PHONY: examples
examples: $(BUILD_DIR)/$(LIB_NAME) $(EXAMPLE_BINS)
	@echo "$(GREEN)✓ All examples built!$(RESET)"

# Build individual example executables
$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(BUILD_DIR)/$(LIB_NAME) | $(BIN_DIR)
	@echo "$(YELLOW)Building example: $@...$(RESET)"
	$(CC) $(CFLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lcortex $(LDFLAGS) -o $@

# Debug build with sanitizers
.PHONY: debug
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean all
	@echo "$(GREEN)✓ Debug build complete (with AddressSanitizer & UBSan)$(RESET)"

# Create build directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "$(GREEN)✓ Clean complete!$(RESET)"

# Show help
.PHONY: help
help:
	@echo "$(GREEN)Cortex Neural Network Framework - Build Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Main targets:$(RESET)"
	@echo "  make              - Build the library (default)"
	@echo "  make all          - Same as above"
	@echo "  make tests        - Build all tests"
	@echo "  make test         - Build and run all tests"
	@echo "  make examples     - Build all examples"
	@echo "  make debug        - Build with debug flags and sanitizers"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "$(YELLOW)Example usage:$(RESET)"
	@echo "  make examples     # Build examples"
	@echo "  ./bin/xor         # Run XOR example"
	@echo "  ./bin/mnist       # Run MNIST example"
	@echo ""
	@echo "$(YELLOW)Directory structure:$(RESET)"
	@echo "  $(BUILD_DIR)/           - Compiled object files and library"
	@echo "  $(BIN_DIR)/             - Test and example executables"
	@echo "  $(SRC_DIR)/             - Source implementation files"
	@echo "  $(INCLUDE_DIR)/         - Public header files"

# Print variables for debugging the Makefile itself
.PHONY: print-vars
print-vars:
	@echo "$(YELLOW)Makefile Variables:$(RESET)"
	@echo "CC        = $(CC)"
	@echo "CFLAGS    = $(CFLAGS)"
	@echo "INCLUDES  = $(INCLUDES)"
	@echo "LDFLAGS   = $(LDFLAGS)"
	@echo "SRCS      = $(SRCS)"
	@echo "OBJS      = $(OBJS)"
	@echo "TEST_BINS = $(TEST_BINS)"
	@echo "EXAMPLE_BINS = $(EXAMPLE_BINS)"

# Dependency tracking (auto-regenerate if headers change)
-include $(OBJS:.o=.d)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "$(YELLOW)Compiling $<...$(RESET)"
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Declare phony targets (targets that don't represent files)
.PHONY: all tests test examples debug clean help print-vars
