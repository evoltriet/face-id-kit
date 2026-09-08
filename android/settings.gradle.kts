pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "face-id-kit-android"
include(":core")
if (!providers.gradleProperty("coreOnly").isPresent) include(":opencv", ":storage")
