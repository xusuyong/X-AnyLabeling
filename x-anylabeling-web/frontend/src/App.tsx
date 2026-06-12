import ProjectSelector from "./components/ProjectSelector";
import AnnotationWorkspace from "./components/AnnotationWorkspace";
import { useProjectStore } from "./stores/projectStore";

export default function App() {
  const { currentProject, selectProject } = useProjectStore();

  if (currentProject) {
    return <AnnotationWorkspace />;
  }

  return <ProjectSelector onSelect={(id) => selectProject(id)} />;
}
