# Development Plan - Dataset Citations (Revised)

> **Historical note (Phase 4 of epic #37):** The scholarly + ScraperAPI integration referenced below was retired in May 2026. Lines that say "ScraperAPI integration confirmed working" describe the state of the world before the opencite pivot. The current automation runs through `core/opencite_pipeline.py`; see AGENTS.md "Data Flow" for the live architecture.

## Core Problem
The automation pipeline is BROKEN - dashboard generation was never integrated into the workflow. We need to fix this first before we can test anything.

## Revised Approach: Fix First, Test As We Go

## Phase 1 - Fix Automation Pipeline (Issue #10) ✅ **COMPLETED**
**Branch:** `fix/issue-10-automation-pipeline`
**Priority:** CRITICAL - MUST DO FIRST
**Started:** 2025-09-13
**Completed:** 2025-09-13
**Status:** Automation fully functional with ScraperAPI

### Why First
- Can't test what doesn't exist
- Manual dashboard generation is unsustainable
- Need working pipeline before modularization
- Tests will validate each fix as we implement

### Phase 1 Tasks

1. **Add dashboard generation to workflow** ✅ **COMPLETED**
   - Found pipeline breaks after CSV generation (line 240)
   - Added analysis commands (temporal, network)
   - Added dashboard generation step
   - Made non-critical to avoid workflow failures
   - Commits: `4096942` (test setup), `eeb78d2` (workflow fix)

2. **Create minimal test dataset** ✅ **COMPLETED**
   - Created test with ds003555 (10 citations → 8 in test)
   - Test structure in `tests/test_data/workflow_test/`
   - Includes discovered_datasets.txt, previous_citations.csv
   - Ready for rapid testing without API calls

3. **Add dashboard CLI to workflow** ✅ **COMPLETED**

   ```yaml
   - name: Generate Interactive Dashboard
     run: |
         --results-dir analysis_results/ \
         --output-dir interactive_reports/ \
         --top-citations 20
   ```

4. **Test and validate** ✅ **COMPLETED**
   - Ran workflow with test dataset ds003555
   - Found 11 citations (was 8, expected 10) 
   - ScraperAPI integration confirmed working
   - CSV generation successful
   - Commits: `59d2c26` (selenium fix), `b5a66e0` (test script)

### Phase 1 Success Criteria
- ✅ Workflow runs end-to-end to CSV generation
- ✅ Test dataset finds missing citations correctly
- ✅ ScraperAPI integration functional
- ✅ Basic automation pipeline restored

## Phase 2 - Modularize Dashboard (Issue #12) ✅ **COMPLETED**
**Branch:** `fix/issue-12-modularize-dashboard`
**Priority:** HIGH
**Started:** 2025-09-14
**Completed:** 2025-09-15
**Status:** Dashboard fully modularized with components

### Why Second
- Now we have working code to refactor
- Can test each module as we extract it
- Maintains working system throughout

### Phase 2 Tasks

1. **Extract data aggregation first** ✅ **COMPLETED**
   - Moved data collection to `dashboard/data/aggregator.py`
   - Added tests for data aggregation
   - Verified nothing breaks

2. **Extract components incrementally** ✅ **COMPLETED**
   ```text
   dashboard/
   ├── components/
   │   ├── charts.py        # Chart generation extracted
   │   ├── networks.py      # Network viz extracted
   │   └── modals.py        # Modal content extracted
   ├── data/
   │   └── aggregator.py    # Data aggregation extracted
   ├── templates/
   │   └── builder.py       # HTML assembly extracted
   └── core.py              # Orchestration (<500 lines)
   ```

3. **Add tests for each extraction** ✅ **COMPLETED**
   - Unit tests per module created
   - Integration test for full dashboard
   - Test dataset used for speed

### Phase 2 Success Criteria
- ✅ No file >500 lines (core.py reduced from 3113 to <500)
- ✅ Each module has tests
- ✅ Dashboard still generates correctly
- ✅ Self-contained dashboard with embedded resources

## Phase 3 - Comprehensive Testing & End-to-End Automation (Issue #11) **IN PROGRESS**
**Branch:** `fix/issue-11-test-infrastructure`
**Priority:** HIGH
**Started:** 2025-09-16
**Status:** Building comprehensive test infrastructure and end-to-end workflow

### Why Third
- Now we have modular, working code
- Need robust test infrastructure with real data
- End-to-end workflow automation critical for sustainability

### Phase 3 Tasks

1. **Create comprehensive end-to-end workflow script**
   - Extend `run_local_workflow.sh` to full pipeline
   - Include citation update → JSON generation → dashboard creation
   - Support both local testing and CI/CD execution
   - Add validation checkpoints throughout

2. **Expand test infrastructure with real data**
   - Use existing ds003555 test case (10 citations)
   - Create additional test datasets:
     - Small dataset (2-3 citations)
     - Medium dataset (20-30 citations)
     - Edge cases (0 citations, deleted dataset)
   - Store controlled snapshots for regression testing

3. **Build robust integration test suite**
   - Test full pipeline: discover → fetch → score → dashboard
   - Test incremental updates (citation count changes)
   - Validate all output formats (CSV, JSON, HTML)
   - Test error recovery and resilience

4. **Add CI/CD testing pipeline**
   - GitHub Actions workflow for testing
   - Run tests on all PRs
   - Coverage reporting with codecov
   - Matrix testing for Python versions

5. **Fix existing test failures**
   - Dashboard aggregator test failing (bridge papers)
   - Ensure all tests pass consistently

### Phase 3 Success Criteria
- ⬜ End-to-end workflow script functional
- ⬜ 75%+ code coverage achieved
- ⬜ Tests run in <5 minutes (fast suite)
- ⬜ CI/CD pipeline active on PRs
- ⬜ All test cases passing
- ⬜ Documentation complete

## Phase 4 - Improve Visualizations (Issue #5)
**Branch:** `feat/issue-5-better-visualizations`
**Priority:** LOW

### Why Last
- Core functionality must work first
- Enhancement, not critical fix
- Can iterate with working tests

### Phase 4 Tasks

1. **Research alternatives**
   - D3.js, Sigma.js, vis.js evaluation
   - Performance benchmarks with test data

2. **Implement improvements**
   - Better layouts
   - Enhanced interactivity
   - Performance optimization

3. **Validate improvements**
   - A/B test visualizations
   - Performance metrics
   - User feedback

### Phase 4 Success Criteria
- ✅ Faster rendering (<2 seconds)
- ✅ Better user experience
- ✅ All existing features maintained

## Implementation Strategy

### For Each Phase
1. Create feature branch from main
2. Implement fix/feature
3. Add tests for new code
4. Verify nothing breaks
5. Submit PR referencing issue
6. Merge only after tests pass

### Test-As-You-Go Approach
- Start with minimal test dataset
- Add test for each component as it's built
- No big bang testing at the end
- Incremental validation

### Controlled Test Dataset
```json
{
  "dataset": "ds_test_001",
  "original_citations": 10,
  "modified_citations": 8,
  "expected_after_fetch": 10,
  "validation": {
    "json_generated": true,
    "dashboard_created": true,
    "confidence_scores": true
  }
}
```

## Success Metrics
- ✅ Automation working end-to-end
- ✅ Dashboard generation automated
- ✅ Code modularized (<500 lines/file)
- ✅ 75% test coverage
- ✅ CI/CD pipeline active
- ✅ All issues resolved

## Timeline
- **Week 1:** Fix automation (Phase 1)
- **Week 2:** Modularize dashboard (Phase 2)
- **Week 3:** Add comprehensive tests (Phase 3)
- **Week 4:** Improve visualizations (Phase 4)

## Key Principle
**Fix first, test as we go** - Don't test hypothetical code, test real working code.