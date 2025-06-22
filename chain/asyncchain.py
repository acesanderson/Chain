from Chain.chain.chain import Chain, Prompt
from Chain.model.model_async import ModelAsync
from Chain.response.response import Response
from Chain.message.message import Message
from Chain.parser.parser import Parser
import asyncio, time
from datetime import datetime
from typing import Optional


class AsyncChain(Chain):

    def __init__(
        self,
        model: ModelAsync,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
    ):
        if not isinstance(model, ModelAsync):
            raise TypeError("Model must be of type ModelAsync")
        if prompt and not isinstance(prompt, Prompt):
            raise TypeError("Prompt must be of type Prompt")
        if parser and not isinstance(parser, Parser):
            raise TypeError("Parser must be of type Parser")
        """Override to use ModelAsync"""
        self.prompt = prompt
        self.model = model
        self.parser = parser
        if self.prompt:
            self.input_schema = self.prompt.input_schema()  # this is a set
        else:
            self.input_schema = set()

    def run(
        self,
        input_variables_list: list[dict] | None = None,
        prompt_strings: list[str] | None = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose=True,
        print_response=False,
    ) -> list[Response]:

        async def _run_async():
            if prompt_strings:
                return await self._run_prompt_strings(
                    prompt_strings, semaphore, cache, verbose, print_response
                )
            if input_variables_list:
                return await self._run_input_variables(
                    input_variables_list, semaphore, cache, verbose, print_response
                )

        results = asyncio.run(_run_async())
        responses = self.convert_results_to_responses(results)

        return responses

    async def _run_prompt_strings(
        self,
        prompt_strings: list[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose=True,
        print_response=False,
    ) -> list:
        """Run multiple prompt strings concurrently with enhanced progress display"""

        # Create concurrent progress tracker if verbose
        tracker = None
        if verbose:
            console = self.model.console or self.__class__._console
            from Chain.progress.tracker import ConcurrentTracker
            from Chain.progress.wrappers import create_concurrent_progress_tracker
            
            tracker = create_concurrent_progress_tracker(console, len(prompt_strings))
            tracker.emit_concurrent_start()

        async def process_with_semaphore_and_tracking(
            prompt_string: str,
            parser: Parser | None,
            semaphore: Optional[asyncio.Semaphore],
            tracker: Optional[ConcurrentTracker],
            cache=True,
            verbose=False,  # Always suppress individual progress during concurrent
            print_response=False,
        ):
            async def do_work():
                if semaphore:
                    async with semaphore:
                        return await self.model.query_async(
                            input=prompt_string,
                            parser=parser,
                            cache=cache,
                            verbose=verbose,
                            print_response=print_response,
                        )
                else:
                    return await self.model.query_async(
                        input=prompt_string,
                        parser=parser,
                        cache=cache,
                        verbose=verbose,
                        print_response=print_response,
                    )
            
            # Wrap with concurrent tracking if available
            if tracker:
                from Chain.progress.wrappers import concurrent_wrapper
                return await concurrent_wrapper(do_work(), tracker)
            else:
                return await do_work()

        # Create coroutines with tracking
        coroutines = [
            process_with_semaphore_and_tracking(
                prompt_string,
                self.parser,
                semaphore,
                tracker,
                cache,
                verbose=False,  # Always suppress individual progress
                print_response=print_response
            )
            for prompt_string in prompt_strings
        ]

        # Run all operations concurrently
        start_time = time.time()
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        duration = time.time() - start_time

        # Complete concurrent tracking
        if tracker:
            tracker.emit_concurrent_complete()

        return results


    async def _run_input_variables(
        self,
        input_variables_list: list[dict],
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose=True,
        print_response=False,
    ) -> list:
        """Run multiple input variable sets concurrently with enhanced progress display"""

        if not self.prompt:
            raise ValueError("No prompt assigned to AsyncChain object")

        # Create concurrent progress tracker if verbose
        tracker = None
        if verbose:
            console = self.model.console or self.__class__._console
            from Chain.progress.tracker import ConcurrentTracker
            from Chain.progress.wrappers import create_concurrent_progress_tracker
            
            tracker = create_concurrent_progress_tracker(console, len(input_variables_list))
            tracker.emit_concurrent_start()

        async def process_with_semaphore_and_tracking(
            input_variables: dict,
            parser: Parser | None,
            semaphore: Optional[asyncio.Semaphore],
            tracker: Optional[ConcurrentTracker],
            cache=True,
            verbose=False,  # Always suppress individual progress during concurrent
            print_response=False,
        ):
            async def do_work():
                if semaphore:
                    async with semaphore:
                        return await self.model.query_async(
                            input=self.prompt.render(input_variables=input_variables),
                            parser=self.parser,
                            cache=cache,
                            verbose=verbose,
                            print_response=print_response,
                        )
                else:
                    return await self.model.query_async(
                        input=self.prompt.render(input_variables=input_variables),
                        parser=self.parser,
                        cache=cache,
                        verbose=verbose,
                        print_response=print_response,
                    )
            
            # Wrap with concurrent tracking if available
            if tracker:
                from Chain.progress.wrappers import concurrent_wrapper
                return await concurrent_wrapper(do_work(), tracker)
            else:
                return await do_work()

        # Create coroutines with tracking
        coroutines = [
            process_with_semaphore_and_tracking(
                input_variables,
                self.parser,
                semaphore,
                tracker,
                cache=cache,
                verbose=False,  # Always suppress individual progress
                print_response=print_response,
            )
            for input_variables in input_variables_list
        ]

        # Run all operations concurrently
        start_time = time.time()
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        duration = time.time() - start_time

        # Complete concurrent tracking
        if tracker:
            tracker.emit_concurrent_complete()

        return results


    def convert_results_to_responses(self, results: list[str]) -> list[Response]:
        # Convert results to Response objects
        responses = []
        for result in results:
            response = Response(
                content=result,
                status="success",
                prompt=None,  # This would be very hard to calculate; maybe later
                model=self.model.model,
                duration=None,  # This would be very hard to calculate; maybe later
                messages=[Message(role="assistant", content=result)],
            )
            responses.append(response)
        return responses
