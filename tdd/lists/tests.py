from django.test import SimpleTestCase

class HomePageTest(SimpleTestCase):

    def test_root_url_resolves_to_home_page_view(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_home_page_returns_correct_html(self):
        response = self.client.get("/")
        self.assertTemplateUsed(response, "home.html")

    def test_home_page_has_todo_in_title(self):
        response = self.client.get("/")
        self.assertContains(response, "To-Do")
