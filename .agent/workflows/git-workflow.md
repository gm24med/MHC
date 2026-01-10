---
description: Git workflow for feature development
---

# Git Workflow

**Always create a new branch for each feature or fix before making changes.**

## Steps

1. **Create a new feature branch** from main:
   ```bash
   git checkout -b feature/descriptive-name
   ```
   Or for fixes:
   ```bash
   git checkout -b fix/descriptive-name
   ```

2. **Make your changes** and commit them:
   ```bash
   git add -A
   git commit -m "type: description"
   ```

3. **Push the branch** to remote:
   ```bash
   git push origin feature/descriptive-name
   ```

4. **Create a Pull Request** on GitHub for review

5. **After PR is merged**, delete the local branch:
   ```bash
   git checkout main
   git pull origin main
   git branch -d feature/descriptive-name
   ```

## Commit Message Convention

Use conventional commits format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
- `ci:` - CI/CD changes

## Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates
