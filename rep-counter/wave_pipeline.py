import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

# Wave pattern definitions for each category
CATEGORY_PATTERNS = {
    1: {(1, 5, 1)},
    2: {(1, 4, 1), (1, 4, 2), (2, 4, 1), (2, 5, 2), (2, 5, 1), (1, 5, 2)},
    3: {(1, 3, 1), (2, 4, 2), (2, 5, 3), (3, 5, 1), (3, 5, 2), (1, 5, 3)},
}

# Priority weights for different wave categories (higher = more important)
PRIORITY_WEIGHT = {1: 1000, 2: 100, 3: 10}

def generate_wave_sequence() -> list[list[int]]:
    """
    Generate predefined wave patterns for testing.
    
    Returns:
        list[list[int]]: List of wave patterns, each as a sequence of integers
    """
    return [
        [1, 2, 3, 4, 5, 4, 3, 2, 1],              # Category 1 pattern
        [1, 2, 1, 2, 3, 4, 3, 2],                # Category 2 pattern
        [2, 3, 4, 5, 4, 3, 2],                   # Category 2 pattern
        [1, 2, 3, 2, 1],                         # Category 3 pattern
        [2, 3, 4, 3, 4, 5, 4, 3, 2, 1],          # Category 2 (2-5-1)
        [3, 4, 5, 4, 3, 2, 1],                   # Category 3 (3-5-1)
    ]

def generate_stream(length: int = 100) -> list[int]:
    """
    Generate a stream of data by concatenating random wave patterns.
    
    Args:
        length (int): Desired length of the generated stream
        
    Returns:
        list[int]: Generated stream of integers between 1 and 5
    """
    sequence = []
    wave_patterns = generate_wave_sequence()
    
    while len(sequence) < length:
        selected_wave = random.choice(wave_patterns)
        
        # Add each value from the selected wave to the sequence
        for current_value in selected_wave:
            if not sequence:
                # First value, add directly
                sequence.append(current_value)
            else:
                last_value = sequence[-1]
                value_difference = current_value - last_value
                
                # Skip if same value
                if value_difference == 0:
                    continue
                # Add directly if difference is 1
                elif abs(value_difference) == 1:
                    sequence.append(current_value)
                else:
                    # Fill gaps with intermediate values
                    direction = 1 if value_difference > 0 else -1
                    current_fill_value = last_value
                    
                    for _ in range(abs(value_difference)):
                        current_fill_value += direction
                        if 1 <= current_fill_value <= 5:
                            sequence.append(current_fill_value)
                        if len(sequence) >= length:
                            break
                            
                if len(sequence) >= length:
                    break
        if len(sequence) >= length:
            break
            
    return sequence[:length]

def reduce_noise(input_array: list[int]) -> list[int]:
    """
    Remove noise patterns from a sequence by detecting and collapsing repeating AB patterns.
    
    Args:
        input_array (list[int]): Input sequence to denoise
        
    Returns:
        list[int]: Denoised sequence with repeating patterns removed
    """
    if len(input_array) < 4:
        return input_array[:]
        
    result = []
    current_index = 0
    
    while current_index < len(input_array):
        result.append(input_array[current_index])
        
        # Check for ABAB pattern (noise pattern)
        if (current_index + 3 < len(input_array) and 
            input_array[current_index] == input_array[current_index + 2] and 
            input_array[current_index + 1] == input_array[current_index + 3]):
            
            # Add the B value
            result.append(input_array[current_index + 1])
            
            # Skip all repeating AB patterns
            skip_index = current_index + 2
            while (skip_index + 1 < len(input_array) and 
                   input_array[skip_index] == input_array[current_index] and 
                   input_array[skip_index + 1] == input_array[current_index + 1]):
                skip_index += 2
            current_index = skip_index
        else:
            current_index += 1
            
    return result

def reduce_noise_iterative(input_array: list[int]) -> list[int]:
    """
    Apply noise reduction iteratively until no more patterns can be removed.
    
    Args:
        input_array (list[int]): Input sequence to denoise
        
    Returns:
        list[int]: Fully denoised sequence
    """
    previous_array = input_array[:]
    
    while True:
        denoised_array = reduce_noise(previous_array)
        # Stop if no more changes
        if len(denoised_array) == len(previous_array):
            break
        previous_array = denoised_array
        
    return denoised_array

def categorize_wave(wave_subsequence: list[int]) -> int | None:
    """
    Classify a wave subsequence into one of the predefined categories.
    
    Args:
        wave_subsequence (list[int]): Sequence representing a potential wave
        
    Returns:
        int | None: Category number (1-3) if valid wave, None otherwise
    """
    sequence_length = len(wave_subsequence)
    if sequence_length < 3:
        return None
        
    # Find the maximum value and its position
    max_value = max(wave_subsequence)
    peak_positions = [i for i, value in enumerate(wave_subsequence) if value == max_value]
    
    # Must have exactly one peak
    if len(peak_positions) != 1:
        return None
        
    peak_index = peak_positions[0]
    
    # Validate strictly increasing before peak
    for i in range(peak_index):
        if wave_subsequence[i + 1] - wave_subsequence[i] != 1:
            return None
            
    # Validate strictly decreasing after peak
    for i in range(peak_index, sequence_length - 1):
        if wave_subsequence[i + 1] - wave_subsequence[i] != -1:
            return None
            
    # Extract start, peak, and end values for pattern matching
    start_value = wave_subsequence[0]
    peak_value = wave_subsequence[peak_index]
    end_value = wave_subsequence[-1]
    
    # Check against known patterns
    for category in sorted(CATEGORY_PATTERNS.keys()):
        if (start_value, peak_value, end_value) in CATEGORY_PATTERNS[category]:
            return category
            
    return None

def segment_waves(sequence: list[int]) -> list[tuple[int, int, int]]:
    """
    Segment a sequence into waves using dynamic programming for optimal categorization.
    
    Args:
        sequence (list[int]): Input sequence to segment
        
    Returns:
        list[tuple[int, int, int]]: List of (start_index, end_index, category) tuples
    """
    sequence_length = len(sequence)
    # Dynamic programming arrays
    dp_scores = [0] * (sequence_length + 1)
    dp_choices = [None] * (sequence_length + 1)
    
    # Fill DP table from right to left
    for start_index in range(sequence_length - 1, -1, -1):
        best_score = 0
        best_choice = None
        
        # Try all possible wave endings
        for end_index in range(start_index + 2, sequence_length):
            wave_subsequence = sequence[start_index:end_index + 1]
            wave_category = categorize_wave(wave_subsequence)
            
            if wave_category is not None:
                current_score = PRIORITY_WEIGHT[wave_category] + dp_scores[end_index]
                if current_score > best_score:
                    best_score = current_score
                    best_choice = (end_index, wave_category)
                    
        dp_scores[start_index] = best_score
        dp_choices[start_index] = best_choice
    
    # Reconstruct the optimal segmentation
    detected_segments = []
    current_index = 0
    
    while current_index < sequence_length and dp_choices[current_index] is not None:
        end_index, wave_category = dp_choices[current_index]
        detected_segments.append((current_index, end_index, wave_category))
        current_index = end_index
        
    return detected_segments

def is_wave_closed(wave_subsequence: list[int]) -> bool:
    """
    Check if a wave subsequence forms a complete, closed wave pattern.
    
    Args:
        wave_subsequence (list[int]): Sequence to check
        
    Returns:
        bool: True if the wave is complete and closed, False otherwise
    """
    sequence_length = len(wave_subsequence)
    if sequence_length < 3:
        return False
        
    max_value = max(wave_subsequence)
    peak_positions = [i for i, value in enumerate(wave_subsequence) if value == max_value]
    
    # Must have exactly one peak
    if len(peak_positions) != 1:
        return False
        
    peak_index = peak_positions[0]
    
    # Peak cannot be at the boundaries
    if peak_index == 0 or peak_index == sequence_length - 1:
        return False
        
    # Validate strictly increasing before peak
    for i in range(peak_index):
        if wave_subsequence[i + 1] - wave_subsequence[i] != 1:
            return False
            
    # Validate strictly decreasing after peak
    for i in range(peak_index, sequence_length - 1):
        if wave_subsequence[i + 1] - wave_subsequence[i] != -1:
            return False
            
    return True

def segment_waves_realtime(sequence: list[int]) -> list[tuple[int, int, int]]:
    """
    Segment waves in real-time, only returning closed/complete waves.
    
    Args:
        sequence (list[int]): Input sequence to segment
        
    Returns:
        list[tuple[int, int, int]]: List of (start_index, end_index, category) tuples
    """
    sequence_length = len(sequence)
    detected_segments = []
    current_index = 0
    
    while current_index < sequence_length - 2:
        wave_found = False
        
        # Try different wave lengths from current position
        for end_index in range(current_index + 2, sequence_length):
            wave_subsequence = sequence[current_index:end_index + 1]
            wave_category = categorize_wave(wave_subsequence)
            
            if wave_category is not None and is_wave_closed(wave_subsequence):
                detected_segments.append((current_index, end_index, wave_category))
                current_index = end_index
                wave_found = True
                break
                
        if not wave_found:
            current_index += 1
            
    return detected_segments

def segment_waves_realtime_improved(sequence: list[int]) -> list[tuple[int, int, int]]:
    """
    Improved real-time wave segmentation with better completion detection.
    
    Args:
        sequence (list[int]): Input sequence to segment
        
    Returns:
        list[tuple[int, int, int]]: List of (start_index, end_index, category) tuples
    """
    if len(sequence) < 3:
        return []
    
    detected_segments = []
    current_index = 0
    sequence_length = len(sequence)
    
    while current_index < sequence_length - 2:
        wave_found = False
        
        # Try different wave lengths with reasonable limits
        for end_index in range(current_index + 2, min(sequence_length, current_index + 15)):
            wave_subsequence = sequence[current_index:end_index + 1]
            
            # Check if this forms a complete wave
            if is_complete_wave_realtime(wave_subsequence):
                wave_category = categorize_wave(wave_subsequence)
                if wave_category is not None:
                    detected_segments.append((current_index, end_index, wave_category))
                    current_index = end_index + 1
                    wave_found = True
                    break
        
        if not wave_found:
            current_index += 1
    
    return detected_segments

def is_complete_wave_realtime(wave_subsequence: list[int]) -> bool:
    """
    Check if a subsequence forms a complete wave suitable for real-time detection.
    
    Args:
        wave_subsequence (list[int]): Sequence to validate
        
    Returns:
        bool: True if the wave is complete and meaningful, False otherwise
    """
    sequence_length = len(wave_subsequence)
    if sequence_length < 3:
        return False
    
    # Find the maximum value and its position
    max_value = max(wave_subsequence)
    max_positions = [i for i, value in enumerate(wave_subsequence) if value == max_value]
    
    # Must have exactly one peak
    if len(max_positions) != 1:
        return False
    
    peak_index = max_positions[0]
    
    # Peak cannot be at the boundaries
    if peak_index == 0 or peak_index == sequence_length - 1:
        return False
    
    # Validate strictly increasing before peak
    for i in range(peak_index):
        if wave_subsequence[i + 1] - wave_subsequence[i] != 1:
            return False
    
    # Validate strictly decreasing after peak
    for i in range(peak_index, sequence_length - 1):
        if wave_subsequence[i + 1] - wave_subsequence[i] != -1:
            return False
    
    # Check for meaningful movement (range should be at least 2)
    wave_range = max_value - min(wave_subsequence)
    if wave_range < 2:
        return False
    
    return True

def detect_waves_streaming_simple(state_buffer: list[int]) -> tuple[list[tuple[int, int, int]], list[int]]:
    """
    Detect completed waves in a streaming buffer and return remaining buffer.
    
    Args:
        state_buffer (list[int]): Current buffer of streaming data
        
    Returns:
        tuple[list[tuple[int, int, int]], list[int]]: (completed_waves, remaining_buffer)
    """
    # Need minimum length for meaningful wave detection
    if len(state_buffer) < 5:
        return [], state_buffer
    
    # Apply noise reduction to clean the buffer
    denoised_buffer = reduce_noise_iterative(state_buffer.copy())
    
    if len(denoised_buffer) < 5:
        return [], state_buffer
    
    # Detect all possible waves in denoised buffer
    all_detected_segments = segment_waves(denoised_buffer)
    
    # Filter for waves that are truly complete (not at buffer end)
    completed_waves = []
    for start_index, end_index, wave_category in all_detected_segments:
        # Only consider waves that end before the buffer end
        if end_index < len(denoised_buffer) - 2:
            completed_waves.append((start_index, end_index, wave_category))
    
    # Determine how much of the buffer to keep
    if completed_waves:
        # Find the end of the last completed wave
        last_wave_end = max(wave[1] for wave in completed_waves)
        # Keep some overlap for continuity
        trim_point = max(0, last_wave_end - 1)
        
        # Map back to original buffer size
        original_trim_point = min(trim_point, len(state_buffer) - 3)
        remaining_buffer = state_buffer[original_trim_point:]
    else:
        # No waves found, manage buffer size
        if len(state_buffer) > 15:
            remaining_buffer = state_buffer[-10:]  # Keep last 10 values
        else:
            remaining_buffer = state_buffer
    
    return completed_waves, remaining_buffer

def plot_pipeline(original_sequence: list[int], denoised_sequence: list[int], detected_patterns: list[tuple[int, int, int]]) -> None:
    """
    Plot the wave detection pipeline results showing original, denoised, and detected patterns.
    
    Args:
        original_sequence (list[int]): Original input sequence
        denoised_sequence (list[int]): Sequence after noise reduction
        detected_patterns (list[tuple[int, int, int]]): Detected wave patterns
    """
    plt.figure(figsize=(16, 8))
    
    # Plot original sequence
    plt.subplot(2, 1, 1)
    plt.plot(original_sequence, label='Original Sequence', color='gray', 
             linestyle='--', marker='o', alpha=0.6)
    plt.title('Original Sequence')
    plt.grid(True)
    
    # Plot denoised sequence with patterns
    plt.subplot(2, 1, 2)
    plt.plot(denoised_sequence, label='Denoised Sequence', color='blue', marker='o')
    
    # Color code different wave categories
    for start_index, end_index, wave_category in detected_patterns:
        if wave_category == 1:
            color = 'green'
        elif wave_category == 2:
            color = 'orange'
        elif wave_category == 3:
            color = 'red'
        else:
            color = 'gray'
        
        plt.axvspan(start_index, end_index, color=color, alpha=0.3, 
                   label=f'Wave Category {wave_category}')
    
    plt.title('Denoised Sequence with Detected Wave Patterns')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Create legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()

def realtime_wave_pipeline(sequence: list[int], animation_interval: float = 0.05) -> list[tuple[int, int, int]]:
    """
    Run real-time wave detection with live visualization.
    
    Args:
        sequence (list[int]): Input sequence to process
        animation_interval (float): Time delay between animation frames
        
    Returns:
        list[tuple[int, int, int]]: Final detected wave segments
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Initialize plot elements
    processing_buffer = []
    line_plot, = ax.plot([], [], color='blue', marker='o', label='Denoised Sequence')
    
    # Set up plot parameters
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Realtime Denoised Sequence with Detected Wave Patterns')
    ax.grid(True)
    
    # Tracking variables
    legend_created = False
    span_artists = []
    last_printed_segments = []
    
    # Process each value in sequence
    for current_index, current_value in enumerate(sequence):
        processing_buffer.append(current_value)
        denoised_sequence = reduce_noise_iterative(processing_buffer)
        
        # Detect all segments and filter for finalized ones
        all_segments = segment_waves(denoised_sequence)
        finalized_segments = [segment for segment in all_segments 
                            if segment[1] < len(denoised_sequence) - 1]
        
        # Update plot line
        line_plot.set_data(range(len(denoised_sequence)), denoised_sequence)
        
        # Remove old wave spans
        for artist in span_artists:
            artist.remove()
        span_artists.clear()
        
        # Draw new wave spans
        for start_index, end_index, wave_category in finalized_segments:
            if wave_category == 1:
                color = 'green'
            elif wave_category == 2:
                color = 'orange'
            elif wave_category == 3:
                color = 'red'
            else:
                color = 'gray'
            
            span = ax.axvspan(start_index, end_index, color=color, alpha=0.3, 
                             label=f'Wave Category {wave_category}')
            span_artists.append(span)
        
        # Print segments only when they change
        if finalized_segments != last_printed_segments:
            print(f"[Realtime] Segments so far: {finalized_segments}")
            last_printed_segments = finalized_segments.copy()
        
        # Create legend once
        if not legend_created:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            legend_created = True
        
        # Update display
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(animation_interval)
    
    plt.ioff()  # Disable interactive mode
    plt.show()
    
    return finalized_segments

def realtime_wave_pipeline_improved(sequence: list[int], animation_interval: float = 0.05) -> list[tuple[int, int, int]]:
    """
    Improved real-time wave pipeline with dual-plot visualization and streaming detection.
    
    Args:
        sequence (list[int]): Input sequence to process
        animation_interval (float): Time delay between animation frames
        
    Returns:
        list[tuple[int, int, int]]: Final detected wave segments
    """
    plt.ion()  # Enable interactive mode
    fig, (original_ax, denoised_ax) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Configure original sequence plot
    original_ax.set_xlim(0, len(sequence))
    original_ax.set_ylim(0.5, 5.5)
    original_ax.set_xlabel('Index')
    original_ax.set_ylabel('Value')
    original_ax.set_title('Original Sequence')
    original_ax.grid(True)
    
    # Configure denoised sequence plot
    denoised_ax.set_xlim(0, len(sequence))
    denoised_ax.set_ylim(0.5, 5.5)
    denoised_ax.set_xlabel('Index')
    denoised_ax.set_ylabel('Value')
    denoised_ax.set_title('Realtime Wave Detection (Denoised)')
    denoised_ax.grid(True)
    
    # Initialize plot lines
    original_line, = original_ax.plot([], [], color='gray', marker='o', alpha=0.6, label='Original')
    denoised_line, = denoised_ax.plot([], [], color='blue', marker='o', label='Denoised')
    
    # Initialize tracking variables
    state_buffer = []
    completed_waves = []
    span_artists = []
    
    # Process each value in the sequence
    for current_index, current_value in enumerate(sequence):
        # Add to processing buffer
        state_buffer.append(current_value)
        
        # Detect newly completed waves
        new_waves, state_buffer = detect_waves_streaming_simple(state_buffer)
        
        # Convert buffer-relative indices to global indices
        for wave_start, wave_end, wave_category in new_waves:
            global_start = current_index - len(state_buffer) - (wave_end - wave_start)
            global_end = global_start + (wave_end - wave_start)
            completed_waves.append((global_start, global_end, wave_category))
        
        # Update original sequence plot
        original_x_data = list(range(current_index + 1))
        original_y_data = sequence[:current_index + 1]
        original_line.set_data(original_x_data, original_y_data)
        
        # Update denoised sequence plot
        if len(state_buffer) >= 3:
            current_denoised = reduce_noise_iterative(state_buffer)
            denoised_start_index = current_index - len(current_denoised) + 1
            denoised_x_data = list(range(denoised_start_index, current_index + 1))
            denoised_y_data = current_denoised
            denoised_line.set_data(denoised_x_data, denoised_y_data)
        
        # Remove old wave visualization spans
        for artist in span_artists:
            artist.remove()
        span_artists.clear()
        
        # Draw completed waves on both plots
        for start_index, end_index, wave_category in completed_waves:
            # Choose color based on category
            if wave_category == 1:
                color = 'green'
                alpha = 0.4
            elif wave_category == 2:
                color = 'orange'
                alpha = 0.4
            elif wave_category == 3:
                color = 'red'
                alpha = 0.4
            else:
                color = 'gray'
                alpha = 0.3
            
            # Draw spans on both plots
            original_span = original_ax.axvspan(start_index, end_index, color=color, alpha=alpha/2)
            denoised_span = denoised_ax.axvspan(start_index, end_index, color=color, alpha=alpha, 
                                              label=f'Wave Cat {wave_category}' if (start_index, end_index, wave_category) == completed_waves[-1] else "")
            span_artists.extend([original_span, denoised_span])
        
        # Print newly detected waves
        if new_waves:
            for wave_start, wave_end, wave_category in new_waves:
                adjusted_start = current_index - len(state_buffer) - (wave_end - wave_start)
                adjusted_end = adjusted_start + (wave_end - wave_start)
                print(f"[Realtime] Wave detected: Cat {wave_category} at indices {adjusted_start}-{adjusted_end}")
        
        # Update legends
        original_ax.legend()
        denoised_ax.legend()
        
        # Refresh display
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(animation_interval)
    
    plt.ioff()  # Disable interactive mode
    plt.show()
    
    # Print final results summary
    print(f"\nFinal Results: {len(completed_waves)} waves detected")
    for wave_number, (start_index, end_index, wave_category) in enumerate(completed_waves):
        print(f"Wave {wave_number + 1}: Category {wave_category}, indices {start_index}-{end_index}")
    
    return completed_waves

if __name__ == "__main__":
    # Generate test sequence
    test_sequence = generate_stream(length=50)
    print("Original sequence:", test_sequence)
    
    # Test improved real-time segmentation
    print("\n--- Testing Improved Realtime Segmentation ---")
    improved_segments = realtime_wave_pipeline_improved(test_sequence, animation_interval=0.1)
    
    # Compare with batch processing
    print("\n--- Batch Segmentation (Reference) ---")
    denoised_test_sequence = reduce_noise_iterative(test_sequence)
    batch_segments = segment_waves(denoised_test_sequence)
    print(f"[Batch] Segments: {batch_segments}")
    
    # Performance comparison
    print("\n--- Comparison ---")
    print(f"Improved realtime: {len(improved_segments)} waves")
    print(f"Batch: {len(batch_segments)} waves")
    
    if len(improved_segments) == len(batch_segments):
        print("Wave count MATCHES!")
    else:
        print("Wave count DIFFERS - this is expected in real-time processing")
    
    # Test streaming detection function directly
    print("\n--- Testing Streaming Detection ---")
    streaming_buffer = []
    total_detected_waves = 0
    
    for value_index, current_value in enumerate(test_sequence):
        streaming_buffer.append(current_value)
        detected_waves, streaming_buffer = detect_waves_streaming_simple(streaming_buffer)
        
        if detected_waves:
            for wave_start, wave_end, wave_category in detected_waves:
                total_detected_waves += 1
                print(f"Stream wave {total_detected_waves}: Category {wave_category} at buffer indices {wave_start}-{wave_end}")
    
    print(f"Total streaming waves detected: {total_detected_waves}")
    
    # Final demonstration
    print("\n--- Final Realtime Demonstration ---")
    final_realtime_segments = realtime_wave_pipeline_improved(test_sequence, animation_interval=0.05)
