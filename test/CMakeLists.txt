
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LLHDC_TEST_DEPENDS
  FileCheck count not
  llhdc
)

add_lit_testsuite(check-llhdc "Running the LLHDC regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLHDC_TEST_DEPENDS}
)
set_target_properties(check-llhdc PROPERTIES FOLDER "Tests")

add_lit_testsuites(LLHDC ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LLHDC_TEST_DEPS}
)

add_subdirectory(Examples)
