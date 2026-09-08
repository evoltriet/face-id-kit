plugins { kotlin("jvm") }
kotlin { jvmToolchain(17) }
dependencies { testImplementation(kotlin("test-junit")); testImplementation("com.google.code.gson:gson:2.13.2") }
tasks.test { workingDir(rootProject.projectDir.parentFile) }
