file(GLOB_RECURSE STYLE_SRC
	${PROJECT_SOURCE_DIR}/src/*.cpp
	${PROJECT_SOURCE_DIR}/test/*.cpp
	${PROJECT_SOURCE_DIR}/src/*.hpp
	${PROJECT_SOURCE_DIR}/test/*.hpp
	${PROJECT_SOURCE_DIR}/src/*.h
	${PROJECT_SOURCE_DIR}/test/*.h)

list(REMOVE_ITEM STYLE_SRC "${PROJECT_SOURCE_DIR}/src/json.hpp")

add_custom_target(
	style
	COMMAND clang-format -style=file
	-i ${STYLE_SRC}
	COMMAND git diff --stat)
