import time

class GenerationTimeMonitor:
    def __init__(self, 
                 tokenized_prompt_length: int,
                 tokenized_context_length: int,
                 tokenized_reference_length: int,
                 max_length: int):
        self.current_token_timer = {}
        self.max_length = max_length
        self.generation_time_cost = {}
        self.generation_length = None
        self.tokenized_lengths = {
            "prompt": tokenized_prompt_length,
            "context": tokenized_context_length,
            "reference": tokenized_reference_length,
        }

    def start_record(self, key: str):
        if key in self.current_token_timer:
            raise ValueError("Token generation is already being recorded.")
        self.current_token_timer[key] = time.perf_counter()
    
    def stop_record(self, key: str):
        if key not in self.current_token_timer:
            raise ValueError("Token generation is not being recorded.")
        if key not in self.generation_time_cost:
            self.generation_time_cost[key] = []
        self.generation_time_cost[key].append(time.perf_counter() - self.current_token_timer[key])
        del self.current_token_timer[key]
    
    def get_report(self, generation_length) -> dict[str, any]:
        report = {
            "generation_length": {
                "max": self.max_length,
                "actual": generation_length,
            },
            "tokenized_lengths": self.tokenized_lengths,
        }
        if not self.generation_time_cost:
            raise ValueError("No time cost data is recorded.")

        time_generation_for_all_tokens = {}
        mean_time_generation_per_token = {}
        
        for key, times in self.generation_time_cost.items():
            total_time = sum(times)
            mean_time = total_time / len(times)
            time_generation_for_all_tokens[key] = total_time
            mean_time_generation_per_token[key] = mean_time

        report['time_cost'] = {
                "overall": time_generation_for_all_tokens,
                "mean": mean_time_generation_per_token,
            }
        return report