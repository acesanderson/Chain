Purpose of this subdirectory --
- define ModelSpec and ModelStore classes
- store the json data for the various modelspecs
- test functions that validate that the ModelSpec is correct
- longer term: modelspec discovery script (using perplexity or something else)

Note:
implementation of these classes will require refactoring:
- Model (base class) .model
- Client and subclasses (for validation vs. ModelSpec)
- Various Message classes
