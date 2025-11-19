# Copilot PR Review Instructions

## 1) Primary review goals (in order)
1. Correctness: ensure code implements intended behavior and passes existing tests.
2. Quality: code should be easy to understand and follow best practices for computational biology/data science
3. Readability & maintainability: clear names, small functions, single responsibility.
4. Avoid over-engineering: simple is better, and the priority should be on a lightweight codebase that is easy to work with. Only add additional complexity where necessary
5. Reproducibility: pipeline code must be deterministic (seeded RNGs, pinned IO formats).
6. Test coverage: tests do not need to exhaustively cover every possible corner case, and should focus on most important functionality

## 2) Tests & CI
- All major feature changes should include unit tests under `tests/`.
- Where necessary, integration tests should be included as well 

## 3) How to structure PR reviews
- The first part of the review should focus on a high-level overview of the proposed changes. Does the approach make sense for the task? Are there major problems with how the code is organized?
- If there are big picture issues, provide an overview of how these might be addressed, with pros and cons of existing approach compared to changes. 
- For anything that isn't a major issue, use inline comments for specific lines to suggest improvements. If a replacement is non-trivial, provide a small code snippet (<=50 lines) and explain why it's better.
- Code changes should be focused on what will have maximum impact on the codebase, and major changes should only be suggested if they are warranted. 
