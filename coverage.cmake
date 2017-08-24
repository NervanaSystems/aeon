if(COVERAGE)
  add_custom_target(
	coverage
	COMMAND ${PROJECT_SOURCE_DIR}/gen_cov.sh
	DEPENDS aeon runtest)
else(COVERAGE)
  message("Without COVERAGE flag coverage raport is unavailable")
endif(COVERAGE)
